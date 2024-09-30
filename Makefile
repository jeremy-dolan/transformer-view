DRAW=/Applications/draw.io.app/Contents/MacOS/draw.io
IN_FILE=gpt2.drawio
OUT_FILE=graph
HTML_ZOOM=130
PNG_WIDTH=1800

html2web:
	dev/html2web
	rm -f src/$(IN_FILE).html

# CLI html export not working, use GUI and above recipe
html:
	$(DRAW) -x --enable-plugins -f html -z $(HTML_ZOOM) -o $(OUT_FILE).html $(IN_FILE)

# CLI svg export has a bug with LaTeX
# Issue opened: https://github.com/jgraph/drawio-desktop/issues/1874
# must use GUI for this as well
svg:
	$(DRAW) -x -f svg -o $(OUT_FILE).svg --width 1200 --embed-svg-images $(IN_FILE)

png:
	$(DRAW) -x -f png -o $(OUT_FILE).png --width $(PNG_WIDTH) $(IN_FILE)

clean:
	rm -f $(OUT_FILE).html $(OUT_FILE).svg $(OUT_FILE).png src/$(IN_FILE).html

