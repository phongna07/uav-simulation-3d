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


def test_broad_phase_matches_exhaustive_search():
    rng=np.random.default_rng(7)
    rects=np.column_stack([rng.uniform(-15,12,40),rng.uniform(-15,12,40),
                           rng.uniform(.2,3,40),rng.uniform(.2,3,40)])
    circles=np.column_stack([rng.uniform(-15,15,30),rng.uniform(-15,15,30),rng.uniform(.1,1.5,30)])
    w=world(rects,circles)
    hits=0
    for trial in range(3000):
        n=rng.integers(1,7)
        old=np.column_stack([rng.uniform(-21,21,(n,2)),rng.uniform(-.2,4,n)])
        # Mostly short physics-like steps, sometimes long sweeps and exact repeats.
        scale=[0,.02,.3,8][trial%4]
        new=old+rng.normal(0,scale,(n,3))
        if trial%7==0 and n>1:
            old[1]=old[0]+[.99,0,0]                  # touching pair, a tie at fraction 0
        expected=w.first_hit_exhaustive(old,new)
        assert w.first_hit(old,new)==expected
        hits+=expected is not None
    assert 500<hits<2900                              # both outcomes are exercised


def test_broad_phase_keeps_exact_boundary_contacts():
    w=world([[0,0,1,1]],[[5,5,1]])
    for old,new in (([[-.5,.5,3]],[[-.5,.5,3]]),        # touching the rectangle side
                    ([[5,6.5,3]],[[5,6.5,3]]),          # touching the circle
                    ([[19.5,0,3]],[[19.5,0,3]]),        # touching the boundary
                    ([[0,0,.5]],[[0,0,.5]])):           # touching the ground
        assert w.first_hit(old,new)==w.first_hit_exhaustive(old,new)
        assert w.first_hit(old,new) is not None
