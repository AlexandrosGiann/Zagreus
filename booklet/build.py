"""
Builds the ZAGREVS booklet.

1. Prints zagrevs.html to an A5 reading PDF with headless Chrome or Edge.
2. Imposes the 80 pages on A4 landscape sheets for duplex printing (flip on short edge)
   in five sewn signatures of 16 pages (4 sheets folded together).

Usage: python build.py
Output: ZAGREVS_reading_A5.pdf, ZAGREVS_print_sewing.pdf
"""
import os
import subprocess
import sys
import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'zagrevs.html')
READING = os.path.join(HERE, 'ZAGREVS_reading_A5.pdf')
SEWING = os.path.join(HERE, 'ZAGREVS_print_sewing.pdf')

BROWSERS = [
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    '/usr/bin/google-chrome', '/usr/bin/chromium',
]
PAGES_PER_SIGNATURE = 16
A4_W, A4_H = 841.89, 595.28


def print_html():
    browser = next((b for b in BROWSERS if os.path.exists(b)), None)
    if not browser:
        sys.exit('Chrome or Edge not found')
    url = 'file:///' + SRC.replace('\\', '/')
    subprocess.run([browser, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                    '--virtual-time-budget=5000', '--print-to-pdf=' + READING, url],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def sheet_sides(n_pages):
    """Yields (left, right) page numbers (1-based) for every printed side, in print order.

    Each signature is 4 sheets folded together. Sheet k of a signature starting at page b+1:
      front: b+16-2k | b+1+2k     back: b+2+2k | b+15-2k
    With a flip on the short edge the back side needs no rotation.
    """
    for base in range(0, n_pages, PAGES_PER_SIGNATURE):
        for k in range(PAGES_PER_SIGNATURE // 4):
            yield base + 16 - 2 * k, base + 1 + 2 * k
            yield base + 2 + 2 * k, base + 15 - 2 * k


def sewing_marks(page):
    # Three punch marks on the fold, as in the reference booklet.
    x = A4_W / 2
    for y in (0.262 * A4_H, 0.5 * A4_H, 0.738 * A4_H):
        page.draw_circle((x, y), 1.8, color=(0, 0, 0), width=0.4)
        page.draw_circle((x, y), 0.35, color=(0, 0, 0), fill=(0, 0, 0), width=0)


def impose():
    src = pymupdf.open(READING)
    n = len(src)
    if n % PAGES_PER_SIGNATURE:
        sys.exit(f'{n} pages: expected a multiple of {PAGES_PER_SIGNATURE}')
    out = pymupdf.open()
    half = A4_W / 2
    for left, right in sheet_sides(n):
        page = out.new_page(width=A4_W, height=A4_H)
        page.show_pdf_page(pymupdf.Rect(0, 0, half, A4_H), src, left - 1)
        page.show_pdf_page(pymupdf.Rect(half, 0, A4_W, A4_H), src, right - 1)
        sewing_marks(page)
    out.set_metadata({'title': 'ZAGREVS', 'author': 'Αλέξανδρος Γιαννάκης',
                      'subject': 'Πέντε μοντέλα σε καθαρή JavaScript'})
    out.save(SEWING, deflate=True, garbage=3)
    return n, len(out)


if __name__ == '__main__':
    print_html()
    pages, sides = impose()
    print(f'{pages} A5 pages -> {sides} A4 sides ({sides // 2} sheets, {pages // PAGES_PER_SIGNATURE} signatures)')
    print(READING)
    print(SEWING)
