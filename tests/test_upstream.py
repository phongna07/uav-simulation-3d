import json
import os
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from uav3d.cli import parser
from uav3d.upstream import age_robot,set_tracker_interval,UPSTREAM


def test_cli_defaults():
    args=parser().parse_args(["run"])
    assert args.scenario=="scen1"
    assert args.controller=="ours"
    assert not hasattr(args,"drones") and not hasattr(args,"scheduling")
    assert parser().parse_args(["benchmark"]).controllers==[
        "ours","apf","pure_mpc","cbf_qp_lag","impc_ob_rec"]


def test_scenario_preset_preserved_in_fresh_process():
    source=f'''
import json,numpy as np,yaml
from uav3d.upstream import load,UPSTREAM
cfg,world,starts=load()
raw=yaml.safe_load((UPSTREAM/'scenarios/scen1.yaml').read_text())
assert np.array_equal(starts,raw['starts'])
assert cfg.TIMESTEP==raw['params']['TIMESTEP']
assert cfg.VMAX==raw['params']['VMAX']
assert cfg.CONTROLLER=='ours'
assert cfg.NUM_ROBOT==len(raw['starts'])
assert world.first_hit(starts,starts) is None
assert not [m for level,m in cfg.validate_config() if level=='ERROR']
print(json.dumps(starts.tolist()))
'''
    command=[sys.executable,"-c",source]
    first=subprocess.check_output(command,text=True)
    second=subprocess.check_output(command,text=True)
    assert json.loads(first)==json.loads(second)


def test_prediction_and_fallback_age_by_elapsed_time():
    values=np.arange(6.)[:,None]
    robot=SimpleNamespace(prediction_epoch=0.,states_prediction=values.copy(),
                          controls_prediction=values.copy(),_mpc_X=values.copy(),_mpc_U=values.copy(),
                          _backup=(values.copy(),values.copy()),_bk_i=0,_plan_left=5,
                          simple=SimpleNamespace(left=5,Xp=values.copy(),Up=values.copy()))
    age_robot(robot,.3,.1)
    assert robot.states_prediction[:,0].tolist()==[3,4,5,5,5,5]
    assert robot._bk_i==2  # Fallback advances the remaining one step on invocation.
    assert robot.simple.left==3
    before=robot.states_prediction.copy()
    age_robot(robot,.3,.1)
    np.testing.assert_array_equal(robot.states_prediction,before)


def test_estimator_interval_preserves_state_and_noise_density():
    sys.path.insert(0,str(UPSTREAM))
    from kalman_target import KalmanTargetTracker
    tracker=KalmanTargetTracker(dt=.1,process_noise_std=2.)
    tracker.initialize([3,4])
    set_tracker_interval(tracker,.3)
    assert tracker.F[0,2]==.3
    assert tracker.Q[2,2]==pytest.approx(4*.3**2)
    np.testing.assert_array_equal(tracker.x,[3,4,0,0])
