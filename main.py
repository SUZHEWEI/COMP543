import numpy as np
import re
import math

ids = ["alt.atheism", "comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware",
       "comp.windows.x",
       "misc.forsale", "rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey", "sci.crypt",
       "sci.electronics", "sci.med",
       "sci.space", "soc.religion.christian", "talk.politics.guns", "talk.politics.mideast", "talk.politics.misc",
       "talk.religion.misc"]


# ------------------Task1--------------
def index2CountArray(index):
    count_array = [0 for i in range(20000)]
    index = list(index)
    for pos in index:
        count_array[pos] += 1
    return np.array(count_array)


# read file and create <key(id), value(word)> map
docs = sc.textFile("s3://chrisjermainebucket/comp330_A6/20_news_same_line.txt").filter(lambda x: 'id' in x)
id_words = docs.map(lambda x: (x[x.find('id="') + 4: x.find('" url=')], x[x.find('">') + 2: x.find('</doc>')]))
# remove numbers
id_words = id_words.map(lambda x: (str(x[0]), re.sub('[^a-zA-Z]', ' ', x[1]).lower().split()))

# get top 20000 words with order
words_count = id_words.flatMap(lambda x: ((word, 1) for word in x[1])).reduceByKey(lambda val1, val2: val1 + val2)
# firstly order by counts desc, then word-self
top_words = sorted(words_count.top(20000, key=lambda x: x[1]), key=lambda x: (-x[1], x[0]))
words = [i[0] for i in top_words]

# create <id, array> map
words2pos = sc.parallelize(range(20000)).map(lambda x: (words[x], x))
words2doc = id_words.flatMap(lambda x: ((word, x[0]) for word in x[1]))
doc_pos = words2doc.join(words2pos).map(lambda x: (x[1][0], x[1][1]))
doc_pos = doc_pos.groupByKey()
doc_posCount = doc_pos.map(lambda x: (x[0], index2CountArray(x[1])))

tmp1 = np.array(doc_posCount.lookup("20_newsgroups/comp.graphics/37261")[0])
print(tmp1[tmp1.nonzero()])

tmp2 = np.array(doc_posCount.lookup("20_newsgroups/talk.politics.mideast/75944")[0])
print(tmp2[tmp2.nonzero()])

tmp3 = np.array(doc_posCount.lookup("20_newsgroups/sci.med/58763")[0])
print(tmp3[tmp3.nonzero()])

# --------------Task2---------------------

tf_vector = doc_posCount.map(lambda x: (x[0], np.divide(x[1], np.sum(x[1]))))
# total doc num
doc_num = doc_posCount.count()
words2docpos = words2doc.join(words2pos)
words2docpos_count = words2docpos.map(lambda x: ((x[1][0], x[1][1]), 1)).reduceByKey(lambda val1, val2: val1 + val2)
words2pos_count = words2docpos_count.map(lambda x: (x[0][1], 1)).reduceByKey(lambda val1, val2: val1 + val2)
idf_vector = words2pos_count.map(lambda x: (x[0], (math.log(doc_num / x[1]))))
idf_vector = sorted(idf_vector.collect(), key=lambda x: (x[0]))
idf_vector = np.array([i[1] for i in idf_vector])
tf_idf = tf_vector.map(lambda x: (x[0], np.array(x[1]) * idf_vector))

tmp1 = tf_idf.lookup("20_newsgroups/comp.graphics/37261")[0]
tmp2 = tf_idf.lookup("20_newsgroups/talk.politics.mideast/75944")[0]
tmp3 = tf_idf.lookup("20_newsgroups/sci.med/58763")[0]

print(tmp1[tmp1.nonzero()])
print(tmp2[tmp2.nonzero()])
print(tmp3[tmp3.nonzero()])


# ---------------Task3--------------
def predictLabel(k, text):
    # filter numbers
    doc_words = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    count = [0 for i in range(20000)]
    for i in range(len(doc_words)):
        word = doc_words[i]
        if word in words:
            index = words.index(word)
            count[index] += 1
    count = np.array(count)
    tf_vector = count / count.sum()
    # idf vector is the same
    tfidf = tf_vector * idf_vector
    distances = tf_idf.map(lambda x: (x[0], np.power(x[1] - tfidf, 2).sum()))
    # order
    distances = distances.takeOrdered(k, lambda x: x[1])
    tmp = {}
    for i in range(len(ids)):
        id = ids[i]
        tmp[id] = 0
    for i in range(len(distances)):
        item = distances[i]
        # like "alt.atheism" in ids
        item_id = item[0].split("/")[1]
        tmp[item_id] += 1
    # return the most frequent
    max_num = 0
    category = -1
    keys = list(tmp.keys())
    for i in range(len(keys)):
        key = keys[i]
        if tmp[key] > max_num:
            max_num = tmp[key]
            category = key
    print(category)


predictLabel(10, "Graphics are pictures and movies created using computers – usually referring to image data created by a computer specifically with help from specialized graphical hardware and software. It is a vast and recent area in computer science. The phrase was coined by computer graphics researchers Verne Hudson and William Fetter of Boeing in 1960. It is often abbreviated as CG, though sometimes erroneously referred to as CGI. Important topics in computer graphics include user interface design, sprite graphics, vector graphics, 3D modeling, shaders, GPU design, implicit surface visualization with ray tracing, and computer vision, among others. The overall methodology depends heavily on the underlying sciences of geometry, optics, and physics. Computer graphics is responsible for displaying art and image data effectively and meaningfully to the user, and processing image data received from the physical world. The interaction and understanding of computers and interpretation of data has been made easier because of computer graphics. Computer graphic development has had a significant impact on many types of media and has revolutionized animation, movies, advertising, video games, and graphic design generally.")

predictLabel(10,
             'A deity is a concept conceived in diverse ways in various cultures, typically as a natural or supernatural being considered divine or sacred. Monotheistic religions accept only one Deity (predominantly referred to as God), polytheistic religions accept and worship multiple deities, henotheistic religions accept one supreme deity without denying other deities considering them as equivalent aspects of the same divine principle, while several non-theistic religions deny any supreme eternal creator deity but accept a pantheon of deities which live, die and are reborn just like any other being. A male deity is a god, while a female deity is a goddess. The Oxford reference defines deity as a god or goddess (in a polytheistic religion), or anything revered as divine. C. Scott Littleton defines a deity as a being with powers greater than those of ordinary humans, but who interacts with humans, positively or negatively, in ways that carry humans to new levels of consciousness beyond the grounded preoccupations of ordinary life.')

predictLabel(10,
             'Egypt, officially the Arab Republic of Egypt, is a transcontinental country spanning the northeast corner of Africa and southwest corner of Asia by a land bridge formed by the Sinai Peninsula. Egypt is a Mediterranean country bordered by the Gaza Strip and Israel to the northeast, the Gulf of Aqaba to the east, the Red Sea to the east and south, Sudan to the south, and Libya to the west. Across the Gulf of Aqaba lies Jordan, and across from the Sinai Peninsula lies Saudi Arabia, although Jordan and Saudi Arabia do not share a land border with Egypt. It is the worlds only contiguous Eurafrasian nation. Egypt has among the longest histories of any modern country, emerging as one of the worlds first nation states in the tenth millennium BC. Considered a cradle of civilisation, Ancient Egypt experienced some of the earliest developments of writing, agriculture, urbanisation, organised religion and central government. Iconic monuments such as the Giza Necropolis and its Great Sphinx, as well the ruins of Memphis, Thebes, Karnak, and the Valley of the Kings, reflect this legacy and remain a significant focus of archaeological study and popular interest worldwide. Egypts rich cultural heritage is an integral part of its national identity, which has endured, and at times assimilated, various foreign influences, including Greek, Persian, Roman, Arab, Ottoman, and European. One of the earliest centers of Christianity, Egypt was Islamised in the seventh century and remains a predominantly Muslim country, albeit with a significant Christian minority.')

predictLabel(10,
             'The term atheism originated from the Greek atheos, meaning without god(s), used as a pejorative term applied to those thought to reject the gods worshiped by the larger society. With the spread of freethought, skeptical inquiry, and subsequent increase in criticism of religion, application of the term narrowed in scope. The first individuals to identify themselves using the word atheist lived in the 18th century during the Age of Enlightenment. The French Revolution, noted for its unprecedented atheism, witnessed the first major political movement in history to advocate for the supremacy of human reason. Arguments for atheism range from the philosophical to social and historical approaches. Rationales for not believing in deities include arguments that there is a lack of empirical evidence; the problem of evil; the argument from inconsistent revelations; the rejection of concepts that cannot be falsified; and the argument from nonbelief. Although some atheists have adopted secular philosophies (eg. humanism and skepticism), there is no one ideology or set of behaviors to which all atheists adhere.')

predictLabel(10,
             'President Dwight D. Eisenhower established NASA in 1958 with a distinctly civilian (rather than military) orientation encouraging peaceful applications in space science. The National Aeronautics and Space Act was passed on July 29, 1958, disestablishing NASAs predecessor, the National Advisory Committee for Aeronautics (NACA). The new agency became operational on October 1, 1958. Since that time, most US space exploration efforts have been led by NASA, including the Apollo moon-landing missions, the Skylab space station, and later the Space Shuttle. Currently, NASA is supporting the International Space Station and is overseeing the development of the Orion Multi-Purpose Crew Vehicle, the Space Launch System and Commercial Crew vehicles. The agency is also responsible for the Launch Services Program (LSP) which provides oversight of launch operations and countdown management for unmanned NASA launches.')

predictLabel(10,
             'The transistor is the fundamental building block of modern electronic devices, and is ubiquitous in modern electronic systems. First conceived by Julius Lilienfeld in 1926 and practically implemented in 1947 by American physicists John Bardeen, Walter Brattain, and William Shockley, the transistor revolutionized the field of electronics, and paved the way for smaller and cheaper radios, calculators, and computers, among other things. The transistor is on the list of IEEE milestones in electronics, and Bardeen, Brattain, and Shockley shared the 1956 Nobel Prize in Physics for their achievement.')

predictLabel(10,
             'The Colt Single Action Army which is also known as the Single Action Army, SAA, Model P, Peacemaker, M1873, and Colt .45 is a single-action revolver with a revolving cylinder holding six metallic cartridges. It was designed for the U.S. government service revolver trials of 1872 by Colts Patent Firearms Manufacturing Company – todays Colts Manufacturing Company – and was adopted as the standard military service revolver until 1892. The Colt SAA has been offered in over 30 different calibers and various barrel lengths. Its overall appearance has remained consistent since 1873. Colt has discontinued its production twice, but brought it back due to popular demand. The revolver was popular with ranchers, lawmen, and outlaws alike, but as of the early 21st century, models are mostly bought by collectors and re-enactors. Its design has influenced the production of numerous other models from other companies.')

predictLabel(10,
             'Howe was recruited by the Red Wings and made his NHL debut in 1946. He led the league in scoring each year from 1950 to 1954, then again in 1957 and 1963. He ranked among the top ten in league scoring for 21 consecutive years and set a league record for points in a season (95) in 1953. He won the Stanley Cup with the Red Wings four times, won six Hart Trophies as the leagues most valuable player, and won six Art Ross Trophies as the leading scorer. Howe retired in 1971 and was inducted into the Hockey Hall of Fame the next year. However, he came back two years later to join his sons Mark and Marty on the Houston Aeros of the WHA. Although in his mid-40s, he scored over 100 points twice in six years. He made a brief return to the NHL in 1979–80, playing one season with the Hartford Whalers, then retired at the age of 52. His involvement with the WHA was central to their brief pre-NHL merger success and forced the NHL to expand their recruitment to European talent and to expand to new markets.')
