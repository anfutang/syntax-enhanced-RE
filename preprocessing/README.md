### This document presents details about how to generate ready-to-use data for syntax-enhanced models.

*️⃣ We perform two kinds of syntactic parses:

- dependency parse using stanza (with biomedical pacakge='CRAFT'); we chose CRAFT since the corpus it is trained on is more related to our relation extraction corpus; on the website of CRAFT, they list the subject of corpus they used: "~100,000 concept annotations to 7 different biomedical ontologies/terminologies; Chemical Entities of Biological Interest; Cell Ontology; Entrez Gene; Gene Ontology (biological process, cellular component, and molecular function); NCBI Taxonomy; Protein Ontology; Sequence Ontology.
- constituency parse using Berkerly Benepar.

We generate four types of data for each of the following models:

- CE (Chunking-Enhanced)
- CT (Constituency-Tree-Enhanced)
- Late Fusion
- MTS (Multi-Task-Syntax)


*️⃣ Prepare wordpiece-level files
Entity markers are added to infuse positional information about arguments, but using these entity markers may influence the quality of syntactic parse. Therefore, in this preprocessing module: 1) entity markers, BERT's special symbols \[CLS\], \[SEP\] are removed, the cleaned list of words to a syntactic parser; 2) after syntactic parse, reinsert these entity markers and BERT's special symbols into the word list. We achieve the re-insertion by aligning indexes of words between the cleaned list of words and the original list of words.

Reinsertion of entity markers and \[CLS\], \[SEP\] comes with linking these tokens with others in syntax tree.

- For dependency trees: (For Late Fusion) entity markers are linked to the syntactic head of the corresponding entity; \[CLS\] and \[SEP\] are linked to the syntactic root; (For MTS) we do not keep entity markers, \[CLS\] and \[SEP\].
- For constituency trees: Entity markers are inserted at the beginning and the end of corresponding entities; \[CLS\] and \[SEP\] are treated as constituents containing themselves.

⭐ You need to generate a dictionary (.pkl file) containing the following keys (refer to demo.pkl):

- wp_ids: word piece ids of each sentence (with entity markers; \[CLS\] \[SEP\] respectively at the beginning and the end of the sentence); ❗note that we use '@@' as subject entity markers, '$$' as object entity markers, and '¢¢' as subject-object (subject and object entity refers to the same entity mention or subject and object entity spans overlap) entity markers. You must use these entity markers, otherwise the pre-processing will fail. Markers are added at the beginning and the end of entities, e.g. "@@ Argatroban @@ has advantages...of clot-bound $$ thrombin $$".
- wps: word pieces of each sentence.
- labels: labels of each sentence saved as tuple (to handle multi-label cases).
- spans: list of indexes of word pieces that make up a word, i.e. given 'ammonia' (indexed by 2), '##gene' (3), '##s' (4), add \[2,5\] indicating we can retrieve the word pieces of word 'ammoniagenes' by sentence\[2:5\].
- words: words of each sentence (just merge word pieces to words; the result should contain \[CLS\], \[SEP\] and entity markers).

*️⃣ The following script assumes that 3 wordpiece-level files (described above) are respectively generated for train, dev and test set. They are saved under the directory /DATA_FOLDER_NAME/DATASET_NAME/wordpiece_level_files (set your own DATA_FOLDER_NAME and DATASET_NAME). Use the following command to generate all data:
```
cd preprocessing
sh preprocess.sh DATASET_NAME DATA_FOLDER_NAME
```

*️⃣ You can also make a pass using the demo data:
```
cd preprocessing
sh preprocess.sh demo ../demo_data
```

*️⃣ Find data for different types of models:

- (for no-syntax) /DATA_FOLDER_NAME/base_files/DATASET_NAME/{train/dev/test}.pkl 
- (for CE) /DATA_FOLDER_NAME/constituency_files/DATASET_NAME/{train/dev/test}_wp2const.pkl
- (for CT) /DATA_FOLDER_NAME/constituency_files/DATASET_NAME/{train/dev/test}_const_seqs.pkl
- (for Late-Fusion)  /DATA_FOLDER_NAME/dependency_files/DATASET_NAME/{train/dev/test}_adjs.pkl
- (for MTS) /DATA_FOLDER_NAME/constituency_files/DATASET_NAME/{train/dev/test}_probe.pkl

⭐ If no error occurs, generated data can be directly passed to syntax-enhanced models.

❗Since the benepar constituency parser can not accept word piece sequence longer than 512, long sentences may cause errors. Indexes of examples on which errors occur will be collected and saved; but you need to manually check these examples and truncate them to a certain length until the tokenized sentence (tokenized by Benepar not BERT) is shorter than 512.

❗In some cases, the entity marker '@' and '$' may not be unique. when there are extra '@' or '$' in the sentence other than entity markers, errors will occur when generating word-level files (under /DATA_FOLDER_NAME/word_level_files/). You need to manually change infos saved in items 'subj_marker_indexes', 'obj_marker_indexes', and 'index_map'. You can make a pass over your dataset to verify if extra '@' or '$' exists.

