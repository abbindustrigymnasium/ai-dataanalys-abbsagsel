from nltk import Tree


def nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [nltk_tree(child) for child in node.children])
    else:
        return node.orth_

def token_form(tok):
    return "_".join([tok.orth_, tok.tag_])

def find_root(docu):
    for token in docu:
        if token.head is token:
            return token

def tree2_nltk(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(token_form(node), [tree2_nltk(child) for child in node.children])
    else:
        return token_form(node)

def spacy_desc_parser(node):
	nsubj = [w for w in node if w.dep_ == 'nsubj']
	for subject in nsubj:
		numbers = [w for w in subject.lefts if w.dep_ == 'nummod']
		if len(numbers) == 1:
			print('nsubj: {}, action: {}, no: {}'.format(subject.text, subject.head.text, numbers[0].text))

            
# [nltk_tree(sent.root).pretty_print() for sent in doc.sents]
