DRAWIO=/Applications/draw.io.app/Contents/MacOS/draw.io

PROJECT=gpt2
SRC_DIR=src
PUB_DIR=pub-web
DEPLOY_DIR=~/code/jeremydolan.net/live/transformer-view

ZOOM=125
BORDER=10
WIDTH=1800

all: svg html clean publish
svg: $(PUB_DIR)/$(PROJECT).svg
html: $(PUB_DIR)/$(PROJECT).html

# SVG EXPORT
# NB: CLI svg export has a bug with LaTeX rendering, I've opened an issue:
#     https://github.com/jgraph/drawio-desktop/issues/1874
# must use GUI for export: 125% width, 10 border, uncheck all boxes
$(PUB_DIR)/$(PROJECT).svg: $(SRC_DIR)/$(PROJECT).svg
	#$(DRAWIO) -x -f svg -o $(PUB_DIR)/$(PROJECT).svg --width $(WIDTH) $(SRC_DIR)/$(PROJECT).drawio
	mv $(SRC_DIR)/$(PROJECT).svg $(PUB_DIR)/$(PROJECT).svg
	mkdir -p $(DEPLOY_DIR)/old
	cp $(DEPLOY_DIR)/$(PROJECT).svg $(DEPLOY_DIR)/old/$(PROJECT).svg
	cp $(PUB_DIR)/$(PROJECT).svg $(DEPLOY_DIR)/$(PROJECT).svg


# HTML+JS EXPORT
# NB: CLI html export is just not working at all
# must use GUI for export: 125% width, uncheck all except zoom and fit
$(PUB_DIR)/$(PROJECT).html: $(SRC_DIR)/$(PROJECT).html
	#$(DRAWIO) -x -f html -z $(HTML_ZOOM) -o $(OUT_FILE).html $(IN_FILE)
	dev/html2web  PASS ARGS
	# xxx update that script
	rm -f $(SRC_DIR)/$(PROJECT).html
	mkdir -p $(DEPLOY_DIR)/old
	cp $(DEPLOY_DIR)/$(PROJECT).html $(DEPLOY_DIR)/old/$(PROJECT).html
	cp $(PUB_DIR)/$(PROJECT).html $(DEPLOY_DIR)/$(PROJECT).html

clean:
	find . -name .DS_Store -type f -delete

publish:
	cp -R assets $(DEPLOY_DIR)

diff:
	cmp $(DEPLOY_DIR)/$(PROJECT).svg  $(PUB_DIR)/$(PROJECT).svg 
	cmp $(DEPLOY_DIR)/$(PROJECT).html $(PUB_DIR)/$(PROJECT).html
	cmp $(DEPLOY_DIR)/preview.png     $(PUB_DIR)/preview.png    
	diff -u $(DEPLOY_DIR)/index.html  $(PUB_DIR)/index.html || [ $$? -eq 1 ]
	diff -ru $(DEPLOY_DIR)/assets     assets                || [ $$? -eq 1 ]
# the disjunct lets make continue if diff returns 1 (differences found)

#png:
#	$(DRAWIO) -x -f png -o $(OUT_FILE).png --width $(WIDTH) $(IN_FILE)
