import json
import os
import numpy as np
import pytest
from uav3d import collect


def make_run(path, controller="ours", horizon=2., full=150., collision=None, status=None, mtime=None):
    path.mkdir(parents=True)
    n = 2
    time = np.linspace(0,horizon if collision is None else collision["time"],21)
    aircraft = np.zeros((len(time),n,17))
    aircraft[:,0,:3] = [0,0,30]
    aircraft[:,1,:3] = [10,0,30]
    target = np.zeros((len(time),3))
    if status is None:
        status = np.zeros((len(time),n,3),int)
    cfg = dict(scenario="scen1",controller=controller,seed=0,starts=aircraft[0,:,:3].tolist(),
               horizon_s=horizon,full_target_duration_s=full,planning_period=.1,
               scheduling="parallel",compute_ratio=0.,command_execution="hold_latest",
               params=dict(VIEWING_RADIUS=25.,ROBOT_RADIUS=.5),
               scenario_definition=dict(rects=[[50,-5,10,10]],circles=[],waypoints=[[0,0],[100,0]]))
    (path/"config.json").write_text(json.dumps(cfg))
    events = [dict(round=r,drone=i,observed_at=r*.1,started_at=r*.1,completed_at=r*.1,
                   execution_s=.01*(i+1),injected_delay_s=0.,queue_wait_s=0.,latency_s=0.,
                   delivered=True,upstream_stop=False,used_backup=False,solver_failures=r//5)
              for r in range(int(round(horizon/.1))) for i in range(n)]
    (path/"events.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events))
    np.savez_compressed(path/"trajectory.npz",time=time,aircraft=aircraft,target=target,
                        command=np.zeros((len(time),n,3)),status=status,
                        watchdog_braking=np.zeros((len(time),n),bool))
    result = dict(status="collision" if collision else "completed",collision=collision,
                  mission_time_s=float(time[-1]),wall_time_s=1.,throughput=2.,skipped_releases=0)
    (path/"result.json").write_text(json.dumps(result))
    if mtime is not None:
        os.utime(path/"result.json",(mtime,mtime))
    return path


def test_preset_name():
    cfg = dict(scenario="scen1",starts=[[0,0,0]]*5,horizon_s=150.,full_target_duration_s=400.)
    assert collect.preset_name(cfg) == "scen1_150s"
    assert collect.preset_name(cfg|dict(horizon_s=400.)) == "scen1_full_mission"
    assert collect.preset_name(cfg|dict(horizon_s=2.5,starts=[[0,0,0]]*9)) == "scen1_2.5s"


def test_short_mission_marks_paper_metrics_unavailable(tmp_path):
    m = collect.evaluate(make_run(tmp_path/"r"))
    assert m["gather_time_s"] is None and m["reform_pct"] is None
    assert "only 0/2 UAVs reached TRACK" in m["paper_notes"]
    assert "re-formation: no UAV joined" in m["paper_notes"]
    assert m["mission_visibility_pct"] == pytest.approx(100)
    assert m["target_loss_pct"] == pytest.approx(0)
    assert m["exec_mean_ms"] == pytest.approx(15)
    assert m["round_median_ms"] == pytest.approx(20)        # parallel: slowest UAV per round
    assert m["solver_failures"] == 2*(19//5)
    assert m["min_obstacle_clearance_m"] == pytest.approx(40-.5)
    assert m["min_inter_uav_m"] == pytest.approx(10)
    assert m["waypoints_reached"] == "1/2"


def test_gathering_excludes_leader(tmp_path):
    status = np.zeros((21,2,3),int)
    status[2:,0,:2] = 1          # leader, first status at t = 0.1 (dt = 0.05)
    status[5:,1,0] = 1           # follower joins at t = 0.25
    m = collect.evaluate(make_run(tmp_path/"r",horizon=1.,full=1.,status=status))
    assert m["preset"] == "scen1_full_mission"
    assert m["gather_time_s"] == pytest.approx(.25)
    assert m["gather_team_s"] == pytest.approx(.25)
    assert m["reform_pct"] == pytest.approx(0)
    assert m["leader_present_pct"] == pytest.approx(100)


def test_collision_cut(tmp_path):
    hit = dict(time=1.,reason="drone_collision",drones=[0,1])
    m = collect.evaluate(make_run(tmp_path/"r",collision=hit))
    assert m["completion_pct"] == pytest.approx(50)
    assert m["target_loss_pct"] == pytest.approx(50)
    assert m["collision_reason"] == "drone_collision (UAV 0, 1)"
    assert "t_c = 1.00 s" in m["paper_notes"]


def test_rows_replace_only_their_algorithm(tmp_path):
    out = tmp_path/"metrics"
    collect.add_run(make_run(tmp_path/"ours1",mtime=1e9),out)
    collect.add_run(make_run(tmp_path/"apf",controller="apf",mtime=1e9),out)
    preset,algorithm,old,skipped = collect.add_run(make_run(tmp_path/"ours2",mtime=2e9),out)
    assert (preset,algorithm,old["run_id"],skipped) == ("scen1_2s","ours","ours1",None)
    rows = collect.read_rows(out/"scen1_2s.csv")
    assert list(rows) == ["ours","apf"]
    assert rows["ours"]["run_id"] == "ours2" and rows["apf"]["run_id"] == "apf"
    # An older run does not replace the latest one unless forced.
    assert collect.add_run(make_run(tmp_path/"ours0",mtime=5e8),out)[3]
    assert collect.read_rows(out/"scen1_2s.csv")["ours"]["run_id"] == "ours2"
    collect.add_run(tmp_path/"ours0",out,force=True)
    assert collect.read_rows(out/"scen1_2s.csv")["ours"]["run_id"] == "ours0"
    md = (out/"scen1_2s.md").read_text()
    assert md.index("## Paper metrics") < md.index("## Run")
    assert "| Gathering time [s] | – | – |" in md
    from openpyxl import load_workbook
    book = load_workbook(out/"scen1_2s.xlsx")
    assert book.sheetnames[0] == "Paper metrics"
    assert [c.value for c in book["Paper metrics"]["A"][1:3]] == ["ours","apf"]


def test_benchmark_directory_expands(tmp_path):
    make_run(tmp_path/"bench"/"ours")
    make_run(tmp_path/"bench"/"apf",controller="apf")
    assert [p.name for p in collect.expand(tmp_path/"bench")] == ["apf","ours"]
