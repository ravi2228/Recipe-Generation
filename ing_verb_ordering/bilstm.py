import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
from ing_verb_ordering.constants import UNK, START_TAG, END_TAG, TRAIN_FILE, DEV_FILE, TEST_FILE
import matplotlib .pyplot as plt
from .viterbi import build_trellis
from .preproc import load_data, get_all_tags
from .most_common import get_word_to_ix
import pickle
from .evaluation import acc
import random
import re


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()


def prepare_sequence(seq, to_ix, USE_CUDA=True):
    idxs = [to_ix[w] if w in to_ix else to_ix[UNK] for w in seq]
    tensor = torch.LongTensor(idxs)
    if USE_CUDA:
        return Variable(tensor).cuda()
    return Variable(tensor)


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec):
    # calculates log_sum_exp in a stable way
    max_score = vec[0][argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return (max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast))))


class BiLSTM(nn.Module):
    """
    Class for the BiLSTM model tagger
    """
    
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, embeddings=None, USE_CUDA=True):
        super(BiLSTM, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.ix_to_tag = {v:k for k,v in tag_to_ix.items()}
        self.tagset_size = len(tag_to_ix)
        self.use_cuda = USE_CUDA
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.num_layers = 1
        if embeddings is not None:
            self.word_embeds.weight.data.copy_(torch.from_numpy(embeddings))

        # Maps the embeddings of the word into the hidden state
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim //2, bidirectional=True, num_layers=self.num_layers)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(in_features=hidden_dim, out_features=self.tagset_size, bias=True)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # axes semantics are: bidirectinal*num_of_layers, minibatch_size, hidden_dimension
        if self.use_cuda:
            return (Variable(torch.randn(2*self.num_layers, 1, self.hidden_dim // 2)).cuda(),
                Variable(torch.randn(2*self.num_layers, 1, self.hidden_dim // 2)).cuda())
        else:
            return (Variable(torch.randn(2*self.num_layers, 1, self.hidden_dim // 2)),
                    Variable(torch.randn(2*self.num_layers, 1, self.hidden_dim // 2)))
    
    def forward(self, sentence):
        """
        The function obtain the scores for each tag for each of the words in a sentence
        Input:
        sentence: a sequence of ids for each word in the sentence
        Make sure to reshape the embeddings of the words before sending them to the BiLSTM. 
        The axes semantics are: seq_len, mini_batch, embedding_dim
        Output: 
        returns lstm_feats: scores for each tag for each token in the sentence.
        """
        self.hidden = self.init_hidden()
        #print(self.hidden)
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        #print("embed " + str(embeds.shape))

        lstm_new, self.hidden = self.lstm(embeds, self.hidden)
        lstm_new = lstm_new.view(len(sentence), self.hidden_dim)
        lstm_features = self.hidden2tag(lstm_new)
        return lstm_features

    
    def predict(self, sentence):
        """
        this function is used for evaluating the model: 
        Input:
            sentence: a sequence of ids for each word in the sentence
        Outputs:
            Obtains the scores for each token by passing through forward, then passes the scores for each token 
            through a softmax-layer and then predicts the tag with the maximum probability for each token: 
            observe that this is like greedy decoding
        """
        lstm_feats = self.forward(sentence)
        softmax_layer = torch.nn.Softmax(dim=1)
        probs = softmax_layer(lstm_feats)
        idx = argmax(probs)
        tags = [self.ix_to_tag[ix] for ix in idx]
        return tags


class BiLSTM_CRF(BiLSTM):
    """
    Class for the BiLSTM_CRF model: derived from the BiLSTM class
    """
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, embeddings=None, USE_CUDA=False):
        super(BiLSTM_CRF, self).__init__(vocab_size, tag_to_ix, embedding_dim, hidden_dim, embeddings, USE_CUDA)
        
        """
        adding tag transitions scores as a parameter.
        """
        self.tag_to_ix = tag_to_ix
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embeddings = embeddings
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix[START_TAG], :] = -1000000
        self.transitions.data[:, tag_to_ix[END_TAG]] = -1000000
        self.use_cuda = USE_CUDA
    
    def forward_alg(self, feats):
        """
        This is the function for the forward algorithm:
        It works very similar to the viterbi algorithm: except that instead of storing just the maximum prev_tag, 
        you sum up the probability to arrive at the curr_tag
        Use log_sum_exp given above to calculate it a numerically stable way.
        
        inputs:
        - feats: -- the hidden states for each token in the input_sequence. 
                Consider this to be the emission potential of each token for each tag.
        - Make sure to use the self.transitions that is defined to capture the tag-transition probabilities
        
        :returns:
        - alpha: -- a pytorch variable containing the score
        """
        
        init_vec = torch.Tensor(1, self.tagset_size).fill_(-1000000)
        # START_TAG has the max score
        init_vec[0][self.tag_to_ix[START_TAG]] = 0.
        prev_scores = torch.autograd.Variable(init_vec)

        for f in feats:
            alphas=[]
            for next_tag in range(self.tagset_size):
                emit_score = f[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = prev_scores + trans_score + emit_score
                alphas.append(log_sum_exp(next_tag_var))
            prev_scores = torch.cat(alphas).view(1,-1)
        terminal_var = prev_scores + self.transitions[self.tag_to_ix[END_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def score_sentence(self,feats, gold_tags):
        """
        Obtain the probability P(x,y) for the labels in tags using the feats and transition_probabilities.
        Inputs:
        - feats: the hidden state scores for each token in the input sentence. 
                Consider this to be the emission potential of each token for each tag.
        - gold_tags: the gold sequence of tags: obtain the joint-log-likelihood score of the sequence 
                    with the feats and gold_tags.
        :returns:
        - a pytorch variable of the score.
        """
        score = torch.autograd.Variable(torch.Tensor([0]))
        if self.use_cuda:
            tags = torch.cat([Variable(torch.LongTensor([self.tag_to_ix[START_TAG]])).cuda(), gold_tags])
        else:
            tags = torch.cat([Variable(torch.LongTensor([self.tag_to_ix[START_TAG]])), gold_tags])
        for i, f in enumerate(feats):
            score += self.transitions[tags[i + 1], tags[i]] + f[tags[i + 1]]
        end_index = self.tag_to_ix[END_TAG]
        final_tag = tags[tags.data.shape[0]-1]
        trans = self.transitions[end_index, final_tag.item()]
        score = score + trans
        return score
    
    def predict(self, sentence):
        """
        This function predicts the tags by using the viterbi algorithm. You should be calling the viterbi algorithm from here.
        Inputs:
        - feats: the hidden state scores for each token in the input sentence. 
                Consider this to be the emission potential of each token for each tag.
        - gold_tags: the gold sequence of tags
        :returns:
        - the best_path which is a sequence of tags
        """
        lstm_feats = self.forward(sentence).view(len(sentence),-1)
        all_tags = [tag for tag,value in self.tag_to_ix.items()]
        
        #call the viterbi algorithm here
        path_score, best_path = build_trellis(all_tags, self.tag_to_ix, lstm_feats, self.transitions.data)
        return best_path

    def neg_log_likelihood(self, lstm_feats, gold_tags):
        
        """
        This function calculates the negative log-likelihood for the CRF: P(Y|X)
        Inputs: 
        lstm_feats: the hidden state scores for each token in the input sentence. 
        gold_tags: the gold sequence of tags
        :returns:
        score of the neg-log-likelihood for the sentence: 
        You should use the previous functions defined: forward_alg, score_sentence
        """
        forward_score = self.forward_alg(lstm_feats)
        gold_score = self.score_sentence(lstm_feats, gold_tags)
        return forward_score - gold_score

        
def special_loss(viter_path_score,gold_score):
    return viter_path_score - gold_score



def train_model(loss, model, X_tr,Y_tr, word_to_ix, tag_to_ix, X_dv=None, Y_dv = None, num_its=50, status_frequency=10,
               optim_args = {'lr':0.1,'momentum':0},
               param_file = 'best.params', USE_CUDA=True):
    
    #initialize optimizer
    #optimizer = optim.SGD(model.parameters(), **optim_args)
    optimizer = optim.Adam(model.parameters())

    losses=[]
    accuracies_train = []
    accuracies = []
    print('starting the epochs')
    for epoch in range(num_its):
        
        loss_value=0
        count1=0
        for X,Y in zip(X_tr,Y_tr):
            X_tr_var = prepare_sequence(X, word_to_ix, USE_CUDA=USE_CUDA)
            Y_tr_var = prepare_sequence(Y, tag_to_ix, USE_CUDA=USE_CUDA)
            # set gradient to zero
            optimizer.zero_grad()
            
            lstm_feats= model.forward(X_tr_var)
            output = loss(lstm_feats,Y_tr_var)
            
            output.backward()
            optimizer.step()
            loss_value += output.item()
            count1+=1
            
            
        losses.append(loss_value/count1)

        # write parameters if this is the best epoch yet
        acc_val=0
        acc_val_train = 0
        count2 = 0
        count2_train = 0
        for Xtr, Ytr in zip(X_dv, Y_dv):
            X_tr_var = prepare_sequence(Xtr, word_to_ix, USE_CUDA=USE_CUDA)
            Y_tr_var = prepare_sequence(Ytr, tag_to_ix, USE_CUDA=USE_CUDA)
            # run forward on dev data
            Y_hat = model.predict(X_tr_var)

            Yhat = np.array([tag_to_ix[yhat] for yhat in Y_hat])
            Ytr = np.array([tag_to_ix[ytr] for ytr in Ytr])

            # compute dev accuracy
            acc_val_train += (acc(Yhat, Ytr)) * len(Xtr)
            count2_train += len(Xtr)
            # save
        acc_val_train /= count2_train
        accuracies_train.append(acc_val_train)
        if X_dv is not None and Y_dv is not None:
            acc_val = 0
            count2=0
            for Xdv, Ydv in zip(X_dv, Y_dv):
                X_dv_var = prepare_sequence(Xdv, word_to_ix, USE_CUDA=USE_CUDA)
                Y_dv_var = prepare_sequence(Ydv, tag_to_ix, USE_CUDA=USE_CUDA)
                # run forward on dev data
                Y_hat = model.predict(X_dv_var)
                
                Yhat = np.array([tag_to_ix[yhat] for yhat in Y_hat])
                Ydv = np.array([tag_to_ix[ydv] for ydv in Ydv])
                
                # compute dev accuracy
                acc_val += (acc(Yhat,Ydv))*len(Xdv)
                count2 += len(Xdv)
                # save
            acc_val/=count2
            if len(accuracies) == 0 or acc_val > max(accuracies):
                state = {'state_dict':model.state_dict(),
                         'epoch':len(accuracies)+1,
                         'accuracy':acc_val}
                torch.save(state,param_file)
            accuracies.append(acc_val)
        # print status message if desired
        if status_frequency > 0 and epoch % status_frequency == 0:
            print("Epoch "+str(epoch+1)+": Dev Accuracy: "+str(acc_val)+' loss: '+str(loss_value))
    return model, losses, accuracies, accuracies_train
            



def plot_results(losses, accuracies, save_name= 'plot_fig.png'):
    fig,ax = plt.subplots(1,2,figsize=[12,2])
    ax[0].plot(losses)
    ax[0].set_ylabel('loss')
    ax[0].set_xlabel('iteration');
    ax[1].plot(accuracies)
    ax[1].set_ylabel('dev set accuracy')
    ax[1].set_xlabel('iteration');
    plt.savefig(save_name)

def plot_results_with_train_acc(losses, train_accuracies, dev_accuracies, save_name= 'plot_fig.png'):
    fig,ax = plt.subplots(1,2,figsize=[12,2])
    ax[0].plot(losses)
    ax[0].set_ylabel('loss')
    ax[0].set_xlabel('iteration')
    ax[1].plot(train_accuracies, label='train acc')
    ax[1].plot(dev_accuracies, label='dev acc')
    ax[1].set_ylabel('accuracy')
    ax[1].legend()
    ax[1].set_xlabel('iteration')
    plt.savefig(save_name)
    
def obtain_polyglot_embeddings(filename, word_to_ix):
    
    vecs = pickle.load(open(filename,'rb'),encoding='latin1')
    
    vocab = [k for k,v in word_to_ix.items()]
    
    word_vecs={}
    for i,word in enumerate(vecs[0]):
        if word in word_to_ix:
            word_vecs[word] = np.array(vecs[1][i])
    
    word_embeddings = []
    for word in vocab:
        if word in word_vecs:
            embed=word_vecs[word]
        else:
            embed=word_vecs[UNK]
        word_embeddings.append(embed)
    
    word_embeddings = np.array(word_embeddings)
    return word_embeddings


def loadGloveModel(gloveFile, word_to_ix):
    f = open(gloveFile,'r', encoding="utf8")
    lines = f.readlines()
    f.close()
    word_vecs = {}
    for line in lines:
        splitLine = line.split()
        word = splitLine[0]
        #print(word)
        if word in word_to_ix:
            vec = [float(val) for val in splitLine[1:]]
            #vec.extend([random.random() for i in range(14)])
            word_vecs[word] = np.array(vec)
        #word_vecs[word] = embedding

    return word_vecs

def obtain_glove_embeddings(filename, word_to_ix):
    word_vecs = loadGloveModel(filename, word_to_ix)
    vocab = [k for k, v in word_to_ix.items()]
    word_embeddings = []
    word_len = len(word_vecs[vocab[0]])
    for word in vocab:
        if word in word_vecs.keys():
            embed = np.array(word_vecs[word])
        else:
            embed = word_vecs["?????"]
        word_embeddings.append(embed)

    word_embeddings = np.array(word_embeddings)
    return word_embeddings


def loadNorwegianModel(norwegianFile, word_to_ix):
    f = open(norwegianFile,'r', encoding="utf8")
    lines = f.readlines()
    f.close()
    word_vecs = {}
    line_num = 0
    len_vec = 0
    super_first = True
    max_len = -1
    while line_num < len(lines):
        line = lines[line_num]
        splitLine = line.split()
        word = splitLine[1]
        if word in word_to_ix:
            first = True
            found_end = False
            vec_len = 0
            while not found_end:
                num_len = len(re.findall(r"[-+]?\d*\.\d+|\d+", line))
                if first:
                    vec_len += num_len - 1
                    first = False
                elif ']' in line:
                    vec_len += num_len
                    found_end = True
                else:
                    vec_len += num_len
                line_num += 1
                line = lines[line_num]
                splitLine = line.split()
            if max_len < vec_len:
                max_len = vec_len
        else:
            line_num += 1
    line_num = 0
    while line_num < len(lines):
        line = lines[line_num]
        splitLine = line.split()
        word = splitLine[1]
        if word in word_to_ix:
            first = True
            found_end = False
            vec = []
            while not found_end:
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if first:
                    vec.extend([float(num) for num in nums[1:]])
                    first = False
                elif ']' in line:
                    vec.extend([float(num) for num in nums])
                    found_end = True
                else:
                    vec.extend([float(num) for num in nums])
                line_num += 1
                line = lines[line_num]
                splitLine = line.split()
            if len(vec) != max_len:
                vec.extend(list(np.random.rand(max_len-len(vec))))
                word_vecs[word] = np.array(vec)
            else:
                word_vecs[word] = np.array(vec)
            if max_len != len(vec):
                print("stay determined")
        else:
            line_num += 1

    return word_vecs

def obtain_norwegian_embeddings(filename, word_to_ix):
    word_vecs = loadNorwegianModel(filename, word_to_ix)
    vocab = [k for k, v in word_to_ix.items()]
    word_embeddings = []
    word_len = len(word_vecs[vocab[0]])
    for word in vocab:
        if word in word_vecs:
            embed = np.array(word_vecs[word])
        else:
            embed = np.random.rand(word_len)
        word_embeddings.append(embed)
    word_embeddings = np.array(word_embeddings)
    return word_embeddings


if __name__ == '__main__':
    tag_to_ix = {}
    all_tags = get_all_tags(TRAIN_FILE)
    all_tags_dev = get_all_tags(DEV_FILE)
    all_tags_tst = get_all_tags(TEST_FILE)
    all_tags = all_tags.union(all_tags_dev)
    all_tags = all_tags.union(all_tags_tst)
    vocab, word_to_ix = get_word_to_ix(TRAIN_FILE)  # obtains all the words in the file
    for tag in all_tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)
    X_tr, Y_tr = load_data(TRAIN_FILE)
    X_dv, Y_dv = load_data(DEV_FILE)
    torch.manual_seed(765)
    loss = torch.nn.CrossEntropyLoss()
    embedding_dim = 30
    hidden_dim = 30
    model = BiLSTM(len(word_to_ix), tag_to_ix, embedding_dim, hidden_dim)
    model, losses, accuracies = train_model(loss, model, X_tr, Y_tr, word_to_ix, tag_to_ix,
                                                   X_dv, Y_dv, num_its=30, status_frequency=2,
                                                   optim_args={'lr': 0.1, 'momentum': 0}, param_file='best.params')

