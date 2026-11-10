import numpy as np
import pytest
from uav3d.metrics import calculate


def test_time_weights_and_collision_penalty():
    time=np.array([0,.2,1.])
    state=np.zeros((3,1,17))
    target=np.array([[0,0,0],[10,0,0],[10,0,0]])
    status=np.zeros((3,1,3))
    status[:,0,0]=1
    m=calculate(time,state,target,status,2,1,result={"status":"collision","collision":{"time":1}})
    assert m["mission_visibility"]==pytest.approx(10)
    assert m["mission_cov_eff"]==pytest.approx(.5)
    assert m["completion_pct"]==50
    assert not m["success"]


def test_gather_and_reform_on_irregular_samples():
    time=np.array([0,.1,.4,.6,1.])
    aircraft=np.zeros((5,2,17))
    status=np.zeros((5,2,3),int)
    status[1:,0,0]=1
    status[2:,1,0]=1
    status[3,1,0]=0
    m=calculate(time,aircraft,np.zeros((5,3)),status,1,1,result={"status":"completed"})
    assert m["gather_time_team_s"]==pytest.approx(.4)
    assert m["gather_time_mean_s"]==pytest.approx(.25)
    assert m["reform_search_pct"]==pytest.approx(100*.4/1.5)
    assert m["reform_episode_s"]==pytest.approx(.4)


def test_zero_duration_first_collision():
    m=calculate(np.array([0.]),np.zeros((1,1,17)),np.zeros((1,3)),np.ones((1,1,3)),
                1,1,result={"status":"collision","collision":{"time":0}})
    assert m["gathered_uavs"]==0
    assert m["mission_visibility"]==0
    assert m["mission_cov_eff"]==0
