#!/bin/bash
PANDOC="/c/Users/25044/AppData/Local/Microsoft/WinGet/Packages/JohnMacFarlane.Pandoc_Microsoft.Winget.Source_8wekyb3d8bbwe/pandoc-3.10/pandoc.exe"
"$PANDOC" paper_submit.tex -o paper_submit.docx --from=latex --to=docx --number-sections
python format_sci.py
echo "Done: paper_submit.docx"
