import pytest
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils import petri_utils


def _place(net, name):
    place = PetriNet.Place(name)
    net.places.add(place)
    return place


def _transition(net, name, label):
    transition = PetriNet.Transition(name, label)
    net.transitions.add(transition)
    return transition


def _accepting(net, start, end):
    im, fm = Marking(), Marking()
    im[start] = 1
    fm[end] = 1
    return net, im, fm


def flower_model(activities):
    net = PetriNet("flower")
    p = _place(net, "p")
    for activity in activities:
        t = _transition(net, f"t_{activity}", activity)
        petri_utils.add_arc_from_to(p, t, net)
        petri_utils.add_arc_from_to(t, p, net)
    return _accepting(net, p, p)


def branching_finite_model():
    """L(M) = {ab, ac}."""
    net = PetriNet("branching")
    p0, p1, p2 = (_place(net, n) for n in ("p0", "p1", "p2"))
    ta, tb, tc = (_transition(net, f"t_{a}", a) for a in "abc")
    petri_utils.add_arc_from_to(p0, ta, net)
    petri_utils.add_arc_from_to(ta, p1, net)
    for t in (tb, tc):
        petri_utils.add_arc_from_to(p1, t, net)
        petri_utils.add_arc_from_to(t, p2, net)
    return _accepting(net, p0, p2)


def silent_branching_model():
    """Same language as branching_finite_model, routed through silent transitions."""
    net = PetriNet("silent_branching")
    p0, p1, p2, p3, p4 = (_place(net, n) for n in ("p0", "p1", "p2", "p3", "p4"))
    ta = _transition(net, "t_a", "a")
    s1, s2 = _transition(net, "tau_1", None), _transition(net, "tau_2", None)
    tb, tc = _transition(net, "t_b", "b"), _transition(net, "t_c", "c")
    petri_utils.add_arc_from_to(p0, ta, net)
    petri_utils.add_arc_from_to(ta, p1, net)
    petri_utils.add_arc_from_to(p1, s1, net)
    petri_utils.add_arc_from_to(s1, p2, net)
    petri_utils.add_arc_from_to(p1, s2, net)
    petri_utils.add_arc_from_to(s2, p3, net)
    petri_utils.add_arc_from_to(p2, tb, net)
    petri_utils.add_arc_from_to(tb, p4, net)
    petri_utils.add_arc_from_to(p3, tc, net)
    petri_utils.add_arc_from_to(tc, p4, net)
    return _accepting(net, p0, p4)


def looping_model():
    """L(M) = a (b)* c: an infinite language."""
    net = PetriNet("loop")
    p0, p1, p2 = (_place(net, n) for n in ("p0", "p1", "p2"))
    ta, tb, tc = (_transition(net, f"t_{a}", a) for a in "abc")
    petri_utils.add_arc_from_to(p0, ta, net)
    petri_utils.add_arc_from_to(ta, p1, net)
    petri_utils.add_arc_from_to(p1, tb, net)
    petri_utils.add_arc_from_to(tb, p1, net)
    petri_utils.add_arc_from_to(p1, tc, net)
    petri_utils.add_arc_from_to(tc, p2, net)
    return _accepting(net, p0, p2)


def dead_branch_model():
    """L(M) = {ab, ac}, plus a visible 'z' branch that deadlocks."""
    net, im, fm = branching_finite_model()
    p1 = next(p for p in net.places if p.name == "p1")
    dead = _place(net, "dead")
    tz = _transition(net, "t_z", "z")
    petri_utils.add_arc_from_to(p1, tz, net)
    petri_utils.add_arc_from_to(tz, dead, net)
    return net, im, fm


@pytest.fixture
def branching():
    return branching_finite_model()


@pytest.fixture
def looping():
    return looping_model()
