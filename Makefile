DRAWIO=/Applications/draw.io.app/Contents/MacOS/draw.io

PROJECT=gpt2
SRC_DIR=src
PUB_DIR=pub-web
DEPLOY_DIR=~/code/jeremydolan.net/live/transformer-view

ZOOM=125
SCALE=1.25
BORDER=10

all: svg html clean publish
svg: $(PUB_DIR)/$(PROJECT).svg
html: $(PUB_DIR)/$(PROJECT).html

### SVG EXPORT ###
# CLI svg export of LaTeX math was fixed in v24.8.0 but lacks anti-aliasing.
# See my open issue: https://github.com/jgraph/drawio-desktop/issues/1874
# Must manually export in GUI before running this recipe.
# GUI export settings: 125% width, 10 border, uncheck all boxes
$(PUB_DIR)/$(PROJECT).svg: $(SRC_DIR)/$(PROJECT).svg
	#$(DRAWIO) -x -f svg -o $(PUB_DIR)/$(PROJECT).svg --scale $(SCALE) $(SRC_DIR)/$(PROJECT).drawio
	mv $(SRC_DIR)/$(PROJECT).svg $(PUB_DIR)/
	mkdir -p $(DEPLOY_DIR)/old
	mv $(DEPLOY_DIR)/$(PROJECT).svg $(DEPLOY_DIR)/old/
	cp $(PUB_DIR)/$(PROJECT).svg $(DEPLOY_DIR)/

### HTML+JS EXPORT ###
# CLI html export not working at all.
# See: https://github.com/jgraph/drawio-desktop/issues/1902
# Must manually export in GUI before running this recipe.
# GUI export settings: 125% width, uncheck all boxes except 'zoom' and 'fit'
$(PUB_DIR)/$(PROJECT).html: $(SRC_DIR)/$(PROJECT).drawio.html
	#$(DRAWIO) -x -f html -z $(HTML_ZOOM) -o $(OUT_FILE).html $(IN_FILE)
	dev/html2web
	rm -f $(SRC_DIR)/$(PROJECT).drawio.html
	mkdir -p $(DEPLOY_DIR)/old
	cp $(DEPLOY_DIR)/$(PROJECT).html $(DEPLOY_DIR)/old/$(PROJECT).html
	cp $(PUB_DIR)/$(PROJECT).html $(DEPLOY_DIR)/$(PROJECT).html

clean:
	find . -name .DS_Store -type f -delete

publish:
	cp $(PUB_DIR)/preview.png $(DEPLOY_DIR)/preview.png
	cp $(PUB_DIR)/index.html  $(DEPLOY_DIR)/index.html
	cp -R assets $(DEPLOY_DIR)

diff:
	cmp      $(DEPLOY_DIR)/$(PROJECT).svg    $(PUB_DIR)/$(PROJECT).svg 
	cmp      $(DEPLOY_DIR)/$(PROJECT).html   $(PUB_DIR)/$(PROJECT).html
	cmp      $(DEPLOY_DIR)/preview.png $(PUB_DIR)/preview.png
	diff -u  $(DEPLOY_DIR)/index.html  $(PUB_DIR)/index.html || [ $$? -eq 1 ]
	diff -ru $(DEPLOY_DIR)/assets      assets || [ $$? -eq 1 ]
# the disjunct lets make continue if diff returns 1 (differences found)

#png:
#	$(DRAWIO) -x -f png -o $(OUT_FILE).png --width $(WIDTH) $(IN_FILE)
