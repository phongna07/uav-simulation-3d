import json
import os
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from uav3d.cli import parser
from uav3d.upstream import age_robot,UPSTREAM


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


def test_buffers_shift_one_step_per_round_after_the_first():
    values=np.arange(6.)[:,None]
    robot=SimpleNamespace(states_prediction=values.copy(),controls_prediction=values.copy(),
                          _mpc_X=values.copy(),_mpc_U=None,_bk_i=0,_plan_left=5)
    age_robot(robot)                                   # before the first plan: unchanged
    assert robot.states_prediction[:,0].tolist()==[0,1,2,3,4,5]
    robot.planned=True
    age_robot(robot)
    for name in ("states_prediction","controls_prediction","_mpc_X"):
        assert getattr(robot,name)[:,0].tolist()==[1,2,3,4,5,5]  # as upstream updateState
    assert robot._bk_i==0 and robot._plan_left==5      # fallback cursors advance in upstream
