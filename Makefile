DRAW=/Applications/draw.io.app/Contents/MacOS/draw.io
IN_FILE=gpt2.drawio
OUT_FILE=graph
HTML_ZOOM=125
SVG_ZOOM=125
SVG_BORDER=10
PNG_WIDTH=1800
WEB_DIR=../jeremydolan.net/live/transformer-view

all: svg html

# CLI svg export has a bug with LaTeX rendering
# Issue opened: https://github.com/jgraph/drawio-desktop/issues/1874
# must use GUI for export; 125% width, 10 border, uncheck all boxes
#$(DRAW) -x -f svg -o $(OUT_FILE).svg --width 1200 $(IN_FILE)
svg: src/gpt2.svg
	mv src/gpt2.svg web/draft-graph.svg
	cp $(WEB_DIR)/draft-graph.svg $(WEB_DIR)/old/draft-graph.svg
	cp web/draft-graph.svg $(WEB_DIR)/draft-graph.svg

# CLI html export not working at all
# must use GUI for export; 125% width, uncheck all except zoom and fit
#$(DRAW) -x -f html -z $(HTML_ZOOM) -o $(OUT_FILE).html $(IN_FILE)
html: src/$(IN_FILE).html
	dev/html2web
	rm -f src/$(IN_FILE).html
	cp $(WEB_DIR)/draft-graph.html $(WEB_DIR)/old/draft-graph.html
	cp web/draft-graph.html $(WEB_DIR)/draft-graph.html

png:
	$(DRAW) -x -f png -o $(OUT_FILE).png --width $(PNG_WIDTH) $(IN_FILE)
