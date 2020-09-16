import torch


class Metric(object):
    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __eq__(self, other):
        return self.score == other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    def __ne__(self, other):
        return self.score != other

    @property
    def score(self):
        raise AttributeError


class F1Metric():
    def __init__(self, vocab, eps=1e-6):
        self.pt = 0
        self.pred = 0
        self.gold = 0
        self.eps = eps
        self.vocab = vocab

    def __call__(self, predictions, golds):
        for pred, gold in zip(predictions, golds):
            pred_entity = self._get_entity(pred)
            gold_entiry = self._get_entity(gold)
            self.pt += len(pred_entity & gold_entiry)
            self.pred += len(pred_entity)
            self.gold += len(gold_entiry)

    @property
    def score(self):
        precision = self.pt / (self.pred + self.eps)
        recall = self.pt / (self.gold + self.eps)
        f1 = (2 * precision * recall) / (precision + recall + self.eps)
        return f1

    def _get_entity(self, ids):
        '''
        B begin
        I intermediate
        E end
        O other
        S single
        '''
        entity, entities = [], []
        for i, tag in enumerate(self.vocab.id2tag(ids)):
            if tag == 'O':
                position = 'O'
            else:
                position, entity_type = tag.split('-')

            if position == 'B':
                if entity:
                    entities.append(tuple(entity))
                entity = [entity_type, i, 1]
            elif position == 'I':
                if entity:
                    if entity_type == entity[0]:
                        entity[-1] += 1
                    else:
                        entities.append(tuple(entity))
                        entity = [entity_type, i, 1]
                else:
                    entity = [entity_type, i, 1]
            elif position == 'E':
                if entity:
                    if entity_type == entity[0]:
                        entity[-1] += 1
                        entities.append(tuple(entity))
                        entity.clear()
                    else:
                        entities.append(tuple(entity))
                        entity.clear()
                        entities.append((entity_type, i, 1))
                else:
                    entities.append((entity_type, i, 1))
                    entity.clear()
            elif position == 'S':
                if entity:
                    entities.append(tuple(entity))
                    entity.clear()
                entities.append((entity_type, i, 1))
            else:
                if entity:
                    entities.append(tuple(entity))
                entity.clear()
        if entity:
            entities.append(tuple(entity))

        return set(entities)