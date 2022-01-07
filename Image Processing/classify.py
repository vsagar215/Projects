import os
import math

#These first two functions require os operations and so are completed for you
#Completed for you
def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
            dataset.append({'label': label, 'bow': bow})
    return dataset

#Completed for you
def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f,'r',encoding='utf-8') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])

#The rest of the functions need modifications ------------------------------
#Needs modifications
def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    # TODO: add your code here
    
    file = open(filepath, "r", encoding='utf-8')
    file_contents = file.read()
    word_list = file_contents.split("\n")
    file.close()
    
    none_counter = 0
    for word in word_list:
        word = word.strip()
        if word in vocab:
            bow[word] = bow.get(word, 0) + 1
        else:
            none_counter += 1
            bow[None] = none_counter
            
    return bow

# Helper function
def log_prior_P(file_count, total_files, smoothing_factor):
    log_val = math.log(file_count + smoothing_factor) - math.log(total_files + 2*smoothing_factor) # ln(A/B) = ln(A) - ln(B)
    return  log_val

def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1  # smoothing factor
    logprob = {}
    # TODO: add your code here
    file_count = {} 

    for val in training_data: 
        if val['label'] in label_list: 
            key = val['label'] 
            file_count[key] = file_count.get(key, 0) + 1

    total_files = sum(file_count.values()) 

    for i in label_list:
        logprob[i] = log_prior_P(file_count[i], total_files, smooth) 

    return logprob

def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1  # smoothing factor
    word_prob = {}
    # TODO: add your code here
    n_total = 0
    
    for count in vocab:
        word_prob[count] = 0

    word_prob[None] = 0

    for dics in training_data:
        if dics['label'] == label:
            for word in dics['bow']:
                key_list = list(dics['bow'].keys())
                if word in vocab:
                    word_prob[word] += dics['bow'][word]
                elif None in key_list:
                    word_prob[None] += dics['bow'][word]
            n_total += sum(dics['bow'].values())

    for i in word_prob: 
        word_prob[i] = log_cond_P(word_prob[i], n_total, smooth, vocab) 

    return word_prob

# Helper Function
def log_cond_P(word_count, tot_word_count, smoothing_factor, vocab):
    ret_val = math.log(word_count + smoothing_factor*1) - math.log(tot_word_count + smoothing_factor*(len(vocab) + 1))
    return ret_val

##################################################################################
# Needs modifications
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = os.listdir(training_directory)
    # TODO: add your code here
    retval['vocabulary'] = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(retval['vocabulary'], training_directory)
    retval['log prior'] = prior(training_data, label_list)
    retval['log p(w|y=2016)'] = p_word_given_label(retval['vocabulary'], training_data, '2016')
    retval['log p(w|y=2020)'] = p_word_given_label(retval['vocabulary'], training_data, '2020')

    return retval

def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    # TODO: add your code here
    retval['log p(y=2016|x)'] = 0
    retval['log p(y=2020|x)'] = 0

    bow = create_bow(model['vocabulary'], filepath)

    for elem in bow:
       increment_by = model['log p(w|y=2016)'][elem] * bow[elem]
       retval['log p(y=2016|x)'] += increment_by
    
    retval['log p(y=2016|x)'] += model['log prior']['2016']

    for val in bow:
        increment_by = model['log p(w|y=2020)'][val] * bow[val]
        retval['log p(y=2020|x)'] += increment_by
    
    retval['log p(y=2020|x)'] += model['log prior']['2020']

    if max(retval['log p(y=2016|x)'], retval['log p(y=2020|x)']) == retval['log p(y=2016|x)']:
        retval['predicted y'] = '2016'
    else:
        retval['predicted y'] = '2020'

    return retval


