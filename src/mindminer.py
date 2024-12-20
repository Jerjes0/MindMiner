from utils import *

class MindMiner:
    def __init__(self, paper_list, api_key):
        self.paper_list = paper_list
        self.api_key = api_key
        self.abstract_dict = {} # dict of all existing abstracts
        self.paper_graph = {}
        self.failed_papers = []
        self.failed_reference_papers = []
        self.text_corpus = {}
        self.tfidf_dict = {}

    def fetch_paper(self,title):
        try:
            pmcid = get_pmcid_by_title(title)
            metadata = fetch_paper_pmc(pmcid)

            metadata_parsed = parse_bioc_xml(metadata)
            references_to_use = [reference for reference in metadata_parsed['references'] if reference.lower() != 'references']

            return references_to_use, metadata_parsed['abstract']
        except:
            try:
                result = get_elsevier_id_by_title(title,self.api_key)
                paper_xml = fetch_elsevier_paper_by_doi(doi=result["doi"], api_key=self.api_key)
                title_references = extract_reference_titles_with_embedded_elements(paper_xml)
                abstract_elsevier = extract_abstract_from_xml(paper_xml)
                return title_references, abstract_elsevier
            
            except Exception as e:
                print(f'failed for title: {title}, error: {e}')
                return None, None


    def extract_titles(self):
        for title in tqdm(self.paper_list):
            references, abstract = self.fetch_paper(title)

            if references and abstract:
                self.abstract_dict[title] = abstract
                self.paper_graph[title] = list_to_dict(references)

            else:
                self.failed_papers.append(title)
        return 


    def dig(self):
        for main_paper, first_level_reference in tqdm(self.paper_graph.items()):
            for reference, _ in tqdm(first_level_reference.items()):
                references, abstract = self.fetch_paper(reference)

                if references and abstract:
                    self.abstract_dict[reference] = abstract
                    self.paper_graph[main_paper][reference] = list_to_dict(references)
                
                else:
                    self.failed_reference_papers.append(reference)
                    continue
                
    def build_single_corpus(self, reference):
        references = [reference]
        for inner_reference in self.paper_graph[reference].keys():
                try:
                    for _inner_reference in self.paper_graph[reference][inner_reference].keys():
                        references.append(_inner_reference)
                except:
                    continue
        references = list(set(references))

        corpus = [self.abstract_dict[_reference] for _reference in references if _reference in self.abstract_dict.keys()]
        
        self.text_corpus[reference] = corpus
    
    def build_corpus(self):
        for reference in self.paper_graph.keys():
            self.build_single_corpus(reference)
    
    def get_single_tfidf(self, reference):
        tfidf = train_tfidf(self.text_corpus[reference])
        self.tfidf_dict[reference] = tfidf
        # return tfidf
    
    def get_tfidf(self):
        for reference in self.paper_graph.keys():
            tfidf = train_tfidf(self.text_corpus[reference])
        self.tfidf_dict[reference] = tfidf

