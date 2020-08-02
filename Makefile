package_name=sarcasm_detector_news_headline

.PHONY: setup visualize lda run-model test

#################################################################################
# COMMANDS                                                                      #
#################################################################################

clean-build:
	rmdir /s dist
	rmdir /s result
	rmdir /s src\sarcasm_detector_news_headline.egg-info

setup: requirements.txt
	pip install -r requirements.txt
	$(info # Build the sdist)
	python setup.py sdist
	pip install dist/sarcasm_detector_news_headline-0.1.0.tar.gz

run-model:
	sarcasm_detector_news_headline run-model

lda:
	sarcasm_detector_news_headline lda

visualize:
	sarcasm_detector_news_headline visualize

test:
	pytest tests