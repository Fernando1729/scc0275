
require(tm)

news = "../20news-18828"
dataset = VCorpus(DirSource(news, mode = "text", recursive = TRUE))

DTM = DocumentTermMatrix(dataset)

