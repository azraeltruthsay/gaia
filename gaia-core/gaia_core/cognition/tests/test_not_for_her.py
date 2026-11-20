"""Messages plainly for someone else (2026-11-20)."""
from gaia_core.cognition.ambient_interject import might_be_for_her, not_for_her


def test_not_for_her():
    assert not_for_her("Azrael: was there an engineering concept you could quiz her on?")
    assert not_for_her("Azrael: Andy, what do you think of that?")
    assert not_for_her("Nick: GAIA's answers are getting better")
    assert not not_for_her("Andy: GAIA, what is the IBC?")
    assert not not_for_her("Nick: So why did you do it for him but not me when I asked")


def test_might_be_for_her():
    assert not might_be_for_her("Azrael: Anyway, down with society, all hail Roko's Basilisk")
    assert might_be_for_her("Nick: So why did you do it for him but not me when I asked")
    assert might_be_for_her("Andy: does anyone know the 2021 IBC egress rules?")
    assert not might_be_for_her("Azrael: was there an engineering concept you could quiz her on?")
