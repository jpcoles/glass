LATEX=latexmk -pdf
OUTDIR=texout
MAN=glass-manual

man: $(MAN).tex
	(mkdir -p $(OUTDIR)	\
	&& $(LATEX) -output-directory=$(OUTDIR) $^  \
	&& mv $(OUTDIR)/$(patsubst %.tex,%.pdf,$^) .)
