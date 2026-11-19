from gaia_web.link_notes import GIF_NOTE, has_gif_link, with_link_notes


def test_gif_hosts_noted():
    for url in ("https://klipy.com/gifs/ai-baby-gloving", "https://tenor.com/view/x-123",
                "https://media.giphy.com/media/abc/giphy.gif", "look https://x.org/a.gif"):
        assert with_link_notes(url).endswith(GIF_NOTE)


def test_other_links_untouched():
    t = "https://app.leg.wa.gov/wac/default.aspx?cite=196"
    assert not has_gif_link(t) and with_link_notes(t) == t


def test_noted_once():
    t = with_link_notes("https://tenor.com/view/x")
    assert with_link_notes(t).count(GIF_NOTE) == 1
