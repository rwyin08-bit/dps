"""Apply SCI formatting to paper_submit.docx. Run after pandoc build."""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

MATH_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
doc = Document('paper_submit.docx')

# Page
for sec in doc.sections:
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.54)
    sec.bottom_margin = Cm(2.54)
    sec.left_margin = Cm(2.54)
    sec.right_margin = Cm(2.54)

# Normal style
ns = doc.styles['Normal']
ns.font.name = 'Times New Roman'
ns.font.size = Pt(11)
ns.font.color.rgb = RGBColor(0, 0, 0)
ns.paragraph_format.line_spacing = 1.5
ns.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
ns.paragraph_format.first_line_indent = Cm(1.27)

# Heading styles (set on both style AND all runs)
for level, size in [('Heading 1', 12), ('Heading 2', 11)]:
    s = doc.styles[level]
    s.font.name = 'Times New Roman'
    s.font.size = Pt(size)
    s.font.bold = True
    s.font.color.rgb = RGBColor(0, 0, 0)
    s.paragraph_format.first_line_indent = Cm(0)

# Also force heading runs to TNR (pandoc may leave them as None)
for p in doc.paragraphs:
    if 'Heading' in (p.style.name or ''):
        for r in p.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(12) if 'Heading 1' in p.style.name else Pt(11)
            r.font.bold = True

# Title, authors, captions, body text
for p in doc.paragraphs:
    st = p.style.name if p.style else ''
    tx = p.text.strip()

    # Force Title style font
    doc.styles['Title'].font.name = 'Times New Roman'
    doc.styles['Title'].font.size = Pt(16)
    doc.styles['Title'].font.bold = True
    doc.styles['Title'].font.color.rgb = RGBColor(0, 0, 0)
    if st == 'Title':
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        for r in p.runs:
            r.font.size = Pt(16); r.font.bold = True; r.font.name = 'Times New Roman'

    if 'Rongwang' in tx:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        for r in p.runs:
            r.font.size = Pt(12); r.font.name = 'Times New Roman'

    if 'Hefei' in tx or 'WenDa' in tx or 'swzhang' in tx:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        for r in p.runs:
            r.font.size = Pt(9); r.font.name = 'Times New Roman'; r.font.italic = True

    if tx.startswith('Figure ') or tx.startswith('Table '):
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        for r in p.runs:
            r.font.size = Pt(9); r.font.name = 'Times New Roman'
        if p.runs:
            p.runs[0].bold = True

    if 'Heading' not in st and st != 'Title':
        for r in p.runs:
            if r.font.name is None or 'Times' not in str(r.font.name):
                r.font.name = 'Times New Roman'
            if r.font.size is None:
                r.font.size = Pt(11)

    for om in p._element.iter(f'{{{MATH_NS}}}oMath'):
        for mr in om.iter(f'{{{MATH_NS}}}r'):
            rPr = mr.find(f'{{{MATH_NS}}}rPr')
            if rPr is None:
                rPr = OxmlElement('m:rPr'); mr.insert(0, rPr)
            if rPr.find(f'{{{MATH_NS}}}nor') is None:
                rPr.append(OxmlElement('m:nor'))

# Tables (three-line format)
for tbl in doc.tables:
    tblPr = tbl._tbl.find(qn('w:tblPr'))
    if tblPr is None:
        tblPr = OxmlElement('w:tblPr'); tbl._tbl.insert(0, tblPr)
    for b in tblPr.findall(qn('w:tblBorders')):
        tblPr.remove(b)
    borders = OxmlElement('w:tblBorders')
    for e, sz in [('top', '12'), ('bottom', '12'), ('insideH', '4')]:
        b = OxmlElement(f'w:{e}')
        b.set(qn('w:val'), 'single'); b.set(qn('w:sz'), sz)
        b.set(qn('w:space'), '0'); b.set(qn('w:color'), '000000')
        borders.append(b)
    for e in ['left', 'right', 'insideV']:
        b = OxmlElement(f'w:{e}')
        b.set(qn('w:val'), 'none'); b.set(qn('w:sz'), '0')
        b.set(qn('w:space'), '0'); b.set(qn('w:color'), 'auto')
        borders.append(b)
    tblPr.append(borders)
    for i, row in enumerate(tbl.rows):
        for cell in row.cells:
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
                    r.font.name = 'Times New Roman'
                    r.bold = (i == 0)

# References: split into individual numbered entries + format
import re
body_elem = doc.element.body
refs_to_remove = []
refs_to_insert = []  # (position_index, new_paragraph_xml)
for i, p in enumerate(doc.paragraphs):
    t = p.text.strip()
    # Find the merged reference paragraph
    if ('Christie' in t and 'Blunt' in t and 'Chung' in t):
        # Split into individual entries
        entries = re.split(r'(?=[A-Z][a-z]+, [A-Z]\.)', t)
        entries = [e.strip().lstrip('99 ').strip() for e in entries if len(e.strip()) > 30]
        # Mark for removal
        refs_to_remove.append(p)
        # Create new paragraphs for each entry
        for k, entry in enumerate(entries):
            new_p = OxmlElement('w:p')
            pPr = OxmlElement('w:pPr')
            ind_el = OxmlElement('w:ind')
            ind_el.set(qn('w:left'), '720'); ind_el.set(qn('w:hanging'), '360')
            pPr.append(ind_el)
            # Line spacing
            spacing = OxmlElement('w:spacing')
            spacing.set(qn('w:line'), '276'); spacing.set(qn('w:lineRule'), 'auto')
            spacing.set(qn('w:after'), '40')
            pPr.append(spacing)
            new_p.append(pPr)
            new_r = OxmlElement('w:r')
            rPr = OxmlElement('w:rPr')
            sz = OxmlElement('w:sz'); sz.set(qn('w:val'), '20')  # 10pt
            rPr.append(sz)
            fn = OxmlElement('w:rFonts'); fn.set(qn('w:ascii'), 'Times New Roman'); fn.set(qn('w:hAnsi'), 'Times New Roman')
            rPr.append(fn)
            new_r.append(rPr)
            new_t = OxmlElement('w:t')
            new_t.text = f'[{k+1}] {entry}'
            new_t.set(qn('xml:space'), 'preserve')
            new_r.append(new_t)
            new_p.append(new_r)
            # Store for insertion after removal
            refs_to_insert.append(new_p)
        break

# Remove old merged paragraph
for p in refs_to_remove:
    p._element.getparent().remove(p._element)

# Insert new individual entries (append at end of body)
for new_p in refs_to_insert:
    body_elem.append(new_p)

# Fix References heading style - ensure it exists and is formatted
found_heading = False
for p in doc.paragraphs:
    if p.text.strip() == 'References' or 'References' in (p.style.name or ''):
        p.style = doc.styles['Heading 1']
        for r in p.runs:
            r.font.name = 'Times New Roman'; r.font.size = Pt(12); r.font.bold = True
        found_heading = True
        break
# If heading was removed, create one before the first reference entry
if not found_heading:
    body = doc.element.body
    # Find first bib entry
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip().startswith('[1]'):
            # Create heading paragraph before it
            h_p = OxmlElement('w:p')
            h_pPr = OxmlElement('w:pPr')
            h_pStyle = OxmlElement('w:pStyle'); h_pStyle.set(qn('w:val'), 'Heading1')
            h_pPr.append(h_pStyle)
            h_p.append(h_pPr)
            h_r = OxmlElement('w:r')
            h_rPr = OxmlElement('w:rPr')
            h_sz = OxmlElement('w:sz'); h_sz.set(qn('w:val'), '24')  # 12pt
            h_rPr.append(h_sz)
            h_b = OxmlElement('w:b')
            h_rPr.append(h_b)
            h_fn = OxmlElement('w:rFonts'); h_fn.set(qn('w:ascii'), 'Times New Roman')
            h_rPr.append(h_fn)
            h_r.append(h_rPr)
            h_t = OxmlElement('w:t'); h_t.text = 'References'
            h_t.set(qn('xml:space'), 'preserve')
            h_r.append(h_t)
            h_p.append(h_r)
            body.insert(list(body).index(p._element), h_p)
            print('Created References heading')
            break

# Page numbers
for sec in doc.sections:
    fp = sec.footer.paragraphs[0] if sec.footer.paragraphs else sec.footer.add_paragraph()
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for txt, tag in [('', 'begin'), (' PAGE ', ''), ('', 'end')]:
        r = fp.add_run(); r.font.size = Pt(10)
        if tag:
            f = OxmlElement('w:fldChar'); f.set(qn('w:fldCharType'), tag); r._r.append(f)
        elif txt:
            i2 = OxmlElement('w:instrText'); i2.text = txt; r._r.append(i2); r.font.size = Pt(1)

doc.save('paper_submit.docx')
print('SCI formatting applied')
