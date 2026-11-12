"""Use the Nimbus Sans family embedded by the manuscript's R figures."""

from pathlib import Path
import hashlib
import tempfile

import matplotlib as mpl
from matplotlib import font_manager


def _convert_type1(source: Path, target: Path, bold: bool) -> None:
    """Prepare a PDF-compatible TrueType font from local Ghostscript outlines."""
    from fontTools.agl import toUnicode
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.boundsPen import BoundsPen
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.t1Lib import T1Font

    original = T1Font(str(source))
    glyphs = original.getGlyphSet()
    info = original["FontInfo"]
    order = [".notdef"] + [name for name in glyphs if name != ".notdef"]
    outlines, metrics, cmap = {}, {}, {}
    for name in order:
        unicode_text = toUnicode(name)
        if len(unicode_text) == 1:
            cmap[ord(unicode_text)] = name
        glyph = glyphs[name]
        bounds = BoundsPen(glyphs)
        glyph.draw(bounds)
        pen = TTGlyphPen(None)
        glyph.draw(Cu2QuPen(pen, max_err=0.25, reverse_direction=True))
        outlines[name] = pen.glyph()
        metrics[name] = (
            round(glyph.width), round(bounds.bounds[0]) if bounds.bounds else 0
        )

    units_per_em = round(1 / original["FontMatrix"][0])
    _, descent, _, ascent = original["FontBBox"]
    builder = FontBuilder(units_per_em, isTTF=True)
    builder.setupGlyphOrder(order)
    builder.setupCharacterMap(cmap)
    builder.setupGlyf(outlines)
    builder.setupHorizontalMetrics(metrics)
    builder.setupHorizontalHeader(ascent=ascent, descent=descent)
    builder.setupNameTable({
        "familyName": info["FamilyName"],
        "styleName": "Bold" if bold else "Regular",
        "uniqueFontIdentifier": original["FontName"],
        "fullName": info["FullName"],
        "psName": original["FontName"],
        "version": "Version " + info["version"],
        "copyright": info.get("Copyright", info.get("Notice", "")),
    })
    builder.setupOS2(
        sTypoAscender=ascent, sTypoDescender=descent, sTypoLineGap=0,
        usWinAscent=ascent, usWinDescent=-descent,
        usWeightClass=700 if bold else 400, fsSelection=32 if bold else 64,
    )
    builder.setupPost(
        italicAngle=info["ItalicAngle"],
        underlinePosition=info["UnderlinePosition"],
        underlineThickness=info["UnderlineThickness"],
    )
    builder.setupMaxp()
    builder.font["head"].macStyle = 1 if bold else 0
    builder.save(target)


def configure_manuscript_font() -> str:
    """Register regular and bold Nimbus Sans, without downloading any fonts.

    Ghostscript ships these faces in Type 1 format, which modern FreeType builds
    may not read. Cache a TrueType conversion in the system temporary directory;
    converting cubic curves within 0.25 font units also permits valid embedding
    with Matplotlib's ``pdf.fonttype=42``.
    """
    family = "Nimbus Sans"
    faces = [entry for entry in font_manager.fontManager.ttflist
             if entry.name == family and Path(entry.fname).suffix.lower() == ".ttf"]
    has_regular = any(entry.style == "normal" and entry.weight in (400, "normal")
                      for entry in faces)
    has_bold = any(entry.style == "normal" and entry.weight in (700, "bold")
                   for entry in faces)
    if not (has_regular and has_bold):
        roots = [Path(prefix) for prefix in
                 ("/usr/local/share", "/opt/homebrew/share", "/usr/share")]
        directories = [directory for root in roots
                       for directory in root.glob("ghostscript/*/Resource/Font")]
        directories += [root / "ghostscript/fonts" for root in roots]
        directories += [Path("/usr/share/fonts/type1/urw-base35")]
        sources = None
        for directory in directories:
            for suffix in ("", ".t1", ".pfb", ".pfa"):
                candidates = [directory / f"NimbusSans-{style}{suffix}"
                              for style in ("Regular", "Bold")]
                if all(path.is_file() for path in candidates):
                    sources = candidates
                    break
            if sources:
                break
        if sources is None:
            raise RuntimeError(
                "Nimbus Sans regular and bold are required to match the manuscript "
                "figures. Install their TrueType fonts or provide a local Ghostscript "
                "installation with NimbusSans-Regular and NimbusSans-Bold."
            )
        for source, bold in zip(sources, (False, True)):
            digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
            cache = Path(tempfile.gettempdir()) / "ctep-manuscript-fonts" / digest
            cache.mkdir(parents=True, exist_ok=True)
            target = cache / ("NimbusSans-Bold.ttf" if bold else "NimbusSans-Regular.ttf")
            if not target.exists():
                _convert_type1(source, target, bold)
            font_manager.fontManager.addfont(target)
    mpl.rcParams["font.family"] = family
    return family
