import json
import os
import re
import requests
from lxml import etree
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

def fetch_paper_pmc(pmc_id, format = 'xml', encoding = 'unicode'):
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_{format}/{pmc_id}/{encoding}"
    response = requests.get(url)
    if response.ok:
        return response.content
    else:
        raise Exception(f"Failed to fetch PMC paper: {pmc_id}")
    

def parse_bioc_xml(xml_content):
    root = etree.fromstring(xml_content)
    extracted_data = {
        "pmc_id": None,
        "doi": None,
        "title": None,
        "abstract": None,
        "authors": [],
        "sections": [],
        "references": [],
        "keywords": []
    }
    pmc_id = root.xpath("//infon[@key='article-id_pmc']/text()")
    extracted_data["pmc_id"] = pmc_id[0] if pmc_id else None
    doi = root.xpath("//infon[@key='article-id_doi']/text()")
    extracted_data["doi"] = doi[0] if doi else None
    title = root.xpath("//passage[infon[@key='section_type']='TITLE']/text/text()")
    extracted_data["title"] = title[0] if title else None
    abstract = root.xpath("//passage[infon[@key='section_type']='ABSTRACT']/text/text()")
    extracted_data["abstract"] = " ".join(abstract) if abstract else None
    authors = root.xpath("//infon[starts-with(@key, 'name_')]/text()")
    extracted_data["authors"] = authors
    sections = root.xpath("//passage[infon[@key='section_type']='INTRO']/text/text()")
    extracted_data["sections"] = sections
    references = root.xpath("//passage[infon[@key='section_type']='REF']/text/text()")
    extracted_data["references"] = references
    keywords = root.xpath("//infon[@key='keywords']/text()")
    extracted_data["keywords"] = keywords if keywords else []
    return extracted_data

def get_pmcid_by_title(title):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pmc",
        "term": f"{title}",
        "retmode": "json",
        "retmax": 1
    }
    response = requests.get(base_url, params=params)
    if response.ok:
        data = response.json()
        pmcids = data.get("esearchresult", {}).get("idlist", [])
        return 'PMC'+pmcids[0] if pmcids else None
    else:
        raise Exception(f"Error querying PMC: {response.status_code}")
    
def get_elsevier_id_by_title(title, api_key):
    base_url = "https://api.elsevier.com/content/search/sciencedirect"
    headers = {
        "X-ELS-APIKey": api_key,
        "Accept": "application/json"
    }
    params = {
        "query": title,
        "count": 1
    }
    response = requests.get(base_url, headers=headers, params=params)
    if response.ok:
        data = response.json()
        entries = data.get("search-results", {}).get("entry", [])
        if entries:
            doi = entries[0].get("prism:doi", None)
            pii = entries[0].get("pii", None)
            return {"doi": doi, "pii": pii}
        else:
            return None
    else:
        raise Exception(f"Error querying Elsevier: {response.status_code}")

def fetch_elsevier_paper_by_doi(doi, api_key):
    base_url = f"https://api.elsevier.com/content/article/doi/{doi}"
    params = {
        'apiKey': api_key,
        'httpAccept': 'text/xml'
    }
    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            return response.text
        else:
            return f"Failed to fetch the paper. Status Code: {response.status_code}, Response: {response.text}"
    except requests.exceptions.RequestException as e:
        return f"An error occurred while fetching the paper: {e}"

def parse_elsevier_xml(xml_content):
    namespaces = {
        'default': 'http://www.elsevier.com/xml/svapi/article/dtd',
        'ce': 'http://www.elsevier.com/xml/common/dtd',
        'prism': 'http://prismstandard.org/namespaces/basic/2.0/',
        'dc': 'http://purl.org/dc/elements/1.1/',
        'xocs': 'http://www.elsevier.com/xml/xocs/dtd'
    }
    root = ET.fromstring(xml_content)
    metadata = {}
    doi = root.find('default:coredata/prism:doi', namespaces)
    metadata['doi'] = doi.text if doi is not None else None
    title = root.find('default:coredata/dc:title', namespaces)
    metadata['title'] = title.text if title is not None else None
    authors = []
    for author in root.findall('.//ce:author-group/ce:author', namespaces):
        given_name = author.find('ce:given-name', namespaces)
        surname = author.find('ce:surname', namespaces)
        if given_name is not None and surname is not None:
            authors.append(f"{given_name.text} {surname.text}")
    metadata['authors'] = authors
    publication_name = root.find('default:coredata/prism:publicationName', namespaces)
    metadata['publication_name'] = publication_name.text if publication_name is not None else None
    cover_date = root.find('default:coredata/prism:coverDate', namespaces)
    metadata['cover_date'] = cover_date.text if cover_date is not None else None
    abstract_parts = []
    for para in root.findall('.//ce:abstract/ce:para', namespaces):
        abstract_parts.append(para.text)
    metadata['abstract'] = ' '.join(abstract_parts) if abstract_parts else None
    references = []
    for ref in root.findall('.//ce:bibliography/ce:bib-reference', namespaces):
        ref_label = ref.find('ce:label', namespaces)
        ref_text = ref.find('.//ce:source-text', namespaces)
        ref_entry = {
            'label': ref_label.text if ref_label is not None else None,
            'text': ref_text.text if ref_text is not None else None
        }
        references.append(ref_entry)
    metadata['references'] = references
    return metadata

def extract_abstract_from_xml(xml_content):
    root = ET.fromstring(xml_content)
    namespaces = {
        'dc': 'http://purl.org/dc/elements/1.1/'
    }
    abstract_tag = root.find('.//dc:description', namespaces)
    if abstract_tag is not None:
        abstract = abstract_tag.text.strip()
        return abstract
    else:
        return "Abstract not found."

def parse_abstract_to_dict(abstract_text):
    lines = [line.strip() for line in abstract_text.split("\n") if line.strip()]
    abstract_dict = {}
    current_key = None
    for line in lines:
        if line.isupper():
            current_key = line.capitalize()
            abstract_dict[current_key] = ""
        elif current_key:
            abstract_dict[current_key] += (" " if abstract_dict[current_key] else "") + line
    for key in abstract_dict:
        abstract_dict[key] = abstract_dict[key].strip()
    return abstract_dict

def extract_reference_titles_with_embedded_elements(xml_content):
    ns = {
        'ce': 'http://www.elsevier.com/xml/common/dtd',
        'sb': 'http://www.elsevier.com/xml/common/struct-bib/dtd'
    }
    def get_full_text(element):
        if element is None:
            return ""
        text_parts = []
        if element.text:
            text_parts.append(element.text)
        for child in element:
            text_parts.append(get_full_text(child))
            if child.tail:
                text_parts.append(child.tail)
        return ''.join(text_parts).strip()
    tree = ET.ElementTree(ET.fromstring(xml_content))
    titles = []
    for bib_ref in tree.findall('.//ce:bib-reference', ns):
        title_element = bib_ref.find('.//sb:title/sb:maintitle', ns)
        if title_element is not None:
            title = get_full_text(title_element)
            if title:
                titles.append(title)
            else:
                titles.append("[Title Missing]")
        else:
            titles.append("[Title Missing]")
    return [title for title in titles if title != "[Title Missing]"]

def parse_abstract_to_dict(abstract_text):
    parts = abstract_text.replace('  ','').split('\n')
    parts_f = [part for part in parts if part != ' ']
    abstract_dict = {}
    for i in range(0, len(parts_f), 2):
        if i + 1 < len(parts_f):
            abstract_dict[parts_f[i]] = parts_f[i+1]
    return abstract_dict

def save_paper_as_pretty_xml(xml_content, file_path):
    dom = parseString(xml_content)
    pretty_xml = dom.toprettyxml(indent="  ")
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(pretty_xml)

def list_to_dict(input_list):
    return {item: None for item in input_list}

