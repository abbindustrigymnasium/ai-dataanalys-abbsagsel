from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer

my_bot = ChatBot(name='KeiBot', read_only=True, logic_adapters=[
                 'chatterbot.logic.MathematicalEvaluation', 'chatterbot.logic.BestMatch'])

corpus_trainer = ChatterBotCorpusTrainer(my_bot)
corpus_trainer.train('chatterbot.corpus.english')

list_trainer = ListTrainer(my_bot)
for item in (small_talk):
    list_trainer.train(item)
small_talk = ['hi.','helo.']