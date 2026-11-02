import matplotlib.pyplot as plt

from costradar import style


def test_brand_style_applied():
    style.apply_style()
    rc = plt.rcParams
    assert rc["axes.facecolor"].upper() == "#FFFBF6"
    assert rc["figure.facecolor"].upper() == "#FFFBF6"
    assert rc["axes.spines.top"] is False and rc["axes.spines.right"] is False
    assert rc["axes.titlesize"] == 15 and rc["axes.titlelocation"] == "left"
    assert rc["axes.titleweight"] == "semibold"
    assert style.FIGSIZE[0] * style.DPI == 1600 and style.FIGSIZE[1] * style.DPI == 900
    assert {style.CRIMSON, style.DEEP_CRIMSON, style.GOLD, style.SAND} == {"#BE1E3C", "#8E1630", "#B08A4E", "#E3D2B4"}
