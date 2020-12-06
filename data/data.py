import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

UNK_TOKEN = 0
SP_TOKEN = 1
EOS_TOKEN = 2
PAD_TOKEN = 3

FEATURE_SIZE = 183

UNK_VECTOR = [UNK_TOKEN] * FEATURE_SIZE
SP_VECTOR = [SP_TOKEN] * FEATURE_SIZE
EOS_VECTOR = [EOS_TOKEN] * FEATURE_SIZE
PAD_VECTOR = [PAD_TOKEN] * FEATURE_SIZE


class NERDataset(Dataset):
    def __init__(self, raw_data, feature=None, device='cpu'):
        super(NERDataset, self).__init__()
        self.source_list = []
        self.target_list = []

        tag_dict = load_tag_dict()

        max_length = self._get_max_length(raw_data)
        print(max_length)
        print()

        for row in raw_data:
            index, syllables, tags = row.rstrip('\n').split('\t')
            syllables_list = syllables.split()
            morphs_list, pos_list = tag_pos(syllables)
            tags_list = tags.split()

            encoded_syllable_list = []
            if feature:
                for morph in morphs_list:
                    if morph in feature.keys():
                        morph_size = len(morph) if morph != '<SP>' else 1
                        encoded_syllable_list += [feature[morph]] * morph_size
                    else:
                        encoded_syllable_list += UNK_VECTOR * len(morph)

                padding_size = max_length - len(encoded_syllable_list)
                encoded_syllable_list.append(EOS_VECTOR)
                encoded_syllable_list += [PAD_VECTOR] * padding_size

                print(len(encoded_syllable_list))
                self.source_list.append(encoded_syllable_list)
            else:
                # TODO: raw feature generation
                pass

            encoded_tag_list = []
            for tag in tags_list:
                if tag in tag_dict.keys():
                    encoded_tag_list.append(tag_dict[tag])
                else:
                    encoded_tag_list.append(tag_dict['<UNK>'])

            padding_size = max_length - len(encoded_tag_list)
            encoded_tag_list.append(tag_dict['<EOS>'])
            encoded_tag_list += [tag_dict['<PAD>']] * padding_size

            self.target_list.append(encoded_tag_list)

        self.source = torch.tensor(self.source_list)
        self.target = torch.tensor(self.target_list)

    def _get_max_length(self, raw_data):
        max_length = 0
        for row in raw_data:
            index, syllables, tags = row.rstrip('\n').split('\t')
            syllables_list = syllables.split()
            length = len(syllables_list)
            if max_length < length:
                max_length = length
        return max_length

    def __str__(self):
        return 'source: {}, target: {}'.format(self.source.shape, self.target.shape)

    def __len__(self):
        return len(self.syllables)

    def __getitem__(self, idx):
        return {
            'source': self.source[idx],
            'target': self.pos[idx],
        }


print()
train_dataset = NERDataset(raw_train_data[:3], feature)
print(train_dataset)
