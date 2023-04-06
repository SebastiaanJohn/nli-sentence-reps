echo "Downloading SNLI dataset..."
wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip -P data/

echo "Unzipping SNLI dataset..."
unzip data/snli_1.0.zip -d data/
rm data/snli_1.0.zip

echo "Done!"