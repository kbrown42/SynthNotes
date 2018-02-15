# SythNotes - A Method for Generation of Mental Health Notes

### Edmon Begoli, Kris Brown, Everett Rush

# Introduction

# Motivation for Research

The incidence of suicide in American service members represents a major health crisis.  Precision medicine and machine learning offer recent breakthroughs and advances that could be used to better understand and mitigate this crisis.  However, given the sensitive nature of medical records, and particularly that of physicatric patient information, there is a major bottleneck in the available data that be utilized to make progress in the task of suicide detection and prevention.  Our recent introduces a system, SynthNotes, which aims to provide sufficiently realistic synthetic psychiatric notes that can be freely shared with the research community without compromising patient confidentiality.  

# Related Work

<!-- This will largely  depend upon the final outcome of our work with regards to methods.  There are of course generation systems that have been used in production.  I'll assemble all those references and discuss here later. --> 

# Approach

## Phase I - Templating
### Generating notes with canned text templates
The first phase of our synthetic note generation takes the most simplistic approach of using a simple template based approach.  We use canned text containing placeholders for variables commonly found in sample SOAP notes and from note guidelines [reference to guidelines].  We also include keywords and life stressor indicators found to be effective as suicide risk indicators.  Most fields are selected from a configuration file by random selection although some are held constant to preserve basic logical coherence.  

## Phase II

### Classic NLG System
Phase II of the synthetic note generation process will incorporate classical NLG techniques.  Due to limitations in data access this is a necessary intermediary step.  There are two areas for improvement which will be discussed below.  
### 1. Patient, Disease, and Physician Modeling 

#### Patient Modeling


Our first step in enriching the synthetic notes model will be patient modeling.  The primary goal of this is to create patients with richer informational properties that can then be reported in the synthetic notes.  This is important as a key goal of our work is to enable search and patient modeling that is useful to the research community at large.  As such, we will be continually enhancing the scope and copmlexity of our synthetic patient models so that they maintain an increasing fidelity to the underlying sensitive medical records of the VA. 

The early stages of patient modeling will entail creating patient profiles that are logically coherent.  Initial properties will cover areas such as demographic information, living conditions, occupation, socioeconomic factors, military service history, marital status, weapon ownership, and some personal narrative information.  The narrative information, as will all other properties will be informed by past research into suicide risk factors.  We envision this to initially provide information on areas such as past criminal history, physical and emotional trauma, history of abuse, etc. 

The set of options for the above mentioned properties will continue to be either hardcoded discrete values or some range or set of values.  It will be the responsibility of some sort of logic engine to search the possible space of options and create logically sounds patient profiles. Logic rules will also be hardcoded.

#### Disease Modeling

Disease modeling is itself a rather large undertaking.  Initial progress can be made by referring to the literature on conditions most commonly associated with suicide risk and focusing on the properties and patient profiles which are present with each disease.  In addition to the corpus of suicide risk research we may also lean upon existing clinical tools such as the UMLS (Unified Medical Language System)[^umls] which provides large dictionaries of medical terms and clusters them into conceptual categories.  Within UMLS is also a semantic net which holds relational information.  We need to adequately explore the capabilities of UMLS.  However, from an initial investigation it appears that a great deal of information regarding relationships of treatments to diseases are encoded in the semantic net.  We can also use the metathesaurus for syntactic variability.  We may also be able use this to add other features to increase the realistic nature of our synthetic notes.  For instance, common abbreviations, misspellings, etc., can be added to notes based on lookups from UMLS.  

A later step, following the initial construction of key disease profiles, would be the logical integration of patient and diseases.  That is, our patients, based on their profile information, should match the probabilistic likelihood of suffering from some disease.  

#### Physician Modeling

Physician modeling refers creating synthetic clinical notes that would be particular to the style of a physician.  As mentioned above, this could contain commono misspellings or usual shorthand or abbreviations that an physician may use.  Given time and a determination of sufficient research value, it could be beneficial to pair semantic meaning with syntactic structure of sentences and explore whether we can find distinct physician groups.  Other language modeling techniques are certainly available such as n-gram modeling.  


#### 2. Classic Sentence Generation

The previously discussed profiles do not discontinue the value of our current template design structure but rather serve to augment it's informational load and logical coherence.  This in turn helps with search methods, and is a useful tool for benchmarking and testing our text processing infrastructure.  However, it is a brittle system that involves a great deal of time in crafting sentences.  The next step our limited data access is to create a sentence generation system.  There is a great deal of literature on this process and we have many options.  However, there is no opensource implementation that is also domain independent.  

Classic sentence generation has typically been described broadly as a three stage pipeline: document planning, microplanning, and surface realization (the actual writing of sentence structures that are present in the final document).  For the beginning of our second phase of SynthNotes, we will be focusing on surface realization from a grammar perspective.  That is, using formal grammars to construct a possible set of legal sentences.  There are of course many different grammars to choose from, but this decision has not yet been set.  

Provided there is a grammar in place, the framework for constructing sentences will involve sets of "messages" that can be constructed with appropriate arguments.  Each message will be given a set of arguments and contain the grammatical structure to write a set of different possible sentences.  For example, we could have a message like PreviousSubstanceAbuse which when given a patient profile will construct a grammatically correct sentence that relates to a patients history of substance abuse.  The most notable opensource tool for surface realization is SimpleNLG[^simplenlg], a Java based tool.  Our current software is in Python, but there are options for integrating the two.  Or, a determination could be made that Java is more suitable because it seems the surface generation of text will consume a large portion of the library's code content.  

Another tool that has been discovered in the last week is the Grammatical Framework (GF)[^gf]. GF is a functional type based language, based on Haskell, that appears to provide tools for sentence construction from a higher level of abstraction, notably the semantic level.  It seems that in some way it is paired with FrameNet[^framenet], a semantically annotated dataset, and could possibly lead to a leaner more concise codebase.  However, the learning curve for this tool seems steep and is an important consideration.  

Given that we are able to successfully create sufficiently realistic notes through the modeling and NLG methods outlined above, we will need to turn to document planning.  This is an area that needs to be explored more by our group.  However, the most common and aggreed upon language structuring theoretical foundation appears to be found in Rhetorical Structure Theory (RST)[^rst1], [^rst2].  This is a natural starting point.  There are other issue related to the second stage of the NLG pipeline, microplanning, that contribute to textual fluency, but at this point are of a smaller concern.  This is especially true if we are assuming search methods that are restricted to the word or phrase level.  


## Phase III
### Data Based Methods
#### 1. Learning templates from a corpus of text

#### 2. Aligning text with knowledge base
#### 3. Concept-to-text generation 

# Study

# Experiments

# Results

# Summary

# References 
[^simplenlg]: SimpleNLG, https://github.com/simplenlg/simplenlg
[^rst1]: Rhetorical Structure Theory, https://en.wikipedia.org/wiki/Rhetorical_structure_theory
[^rst2]: Rhetorical Structure Theory: Toward a functional theory of text organization, https://doi.org/10.1515/text.1.1988.8.3.243
[^gf]: Grammatical Framework, http://www.grammaticalframework.org/
[^framenet]: FrameNet, https://framenet.icsi.berkeley.edu/fndrupal/
[^umls]: Unified Medical Language System, https://www.nlm.nih.gov/research/umls/