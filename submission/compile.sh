cd submission
pdflatex submission.tex
pbibtex submission
pdflatex submission.tex
pdflatex submission.tex
mv submission.* out/
mv out/submission.tex .
cd ..