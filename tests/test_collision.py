import numpy as np
import pytest
from uav3d.collision import CollisionWorld, rectangle_entry


def world(rects=(),circles=()):
    return CollisionWorld(rects,circles,[-20,-20,20,20],.5)


def test_tunnels_through_rectangle():
    hit=world([[0,-1,1,2]]).first_hit([[-2,0,3]],[[3,0,3]])
    assert hit.reason=="obstacle_rectangle"
    assert hit.fraction==pytest.approx(.3)


def test_rectangle_corner_is_rounded_not_box_inflation():
    assert rectangle_entry(np.array([-.45,-.45]),np.zeros(2),[0,0,1,1],.5) is None


def test_circle_tunneling():
    hit=world(circles=[[0,0,1]]).first_hit([[-5,0,3]],[[5,0,3]])
    assert hit.fraction==pytest.approx(.35)


def test_crossing_drones():
    hit=world().first_hit([[-2,0,3],[2,0,3]],[[2,0,3],[-2,0,3]])
    assert hit.reason=="drone_collision"
    assert hit.fraction==pytest.approx(.375)


def test_initial_overlap_and_boundary_ground():
    assert world().first_hit([[0,0,3],[.5,0,3]],[[0,0,3],[.5,0,3]]).fraction==0
    assert world().first_hit([[19,0,3]],[[20,0,3]]).reason=="world_boundary"
    assert world().first_hit([[0,0,1]],[[0,0,0]]).reason=="ground_contact"


def test_earliest_event_and_nonfinite():
    w=world(circles=[[0,0,1]])
    hit=w.first_hit([[-3,0,3]],[[-3,0,-3]])
    assert hit.reason=="ground_contact"
    assert w.first_hit([[0,0,3]],[[np.nan,0,3]]).reason=="non_finite_state"
