# -*- coding: utf-8 -*-
# This file contains the prompts for the function vector analysis.
# It is updated automatically by the translate_prompts.py script.

FUNCTION_TYPES = {
    "abstractive_tasks": [
        "antonym",
        "capitalize",
        "country_capital",
        "country_currency",
        "translation_french",
        "translation_german",
        "translation_spanish",
        "landmark_country",
        "lowercase",
        "national_parks",
        "next_item",
        "previous_item",
        "park_country",
        "person_instrument",
        "person_occupation",
        "person_sport",
        "present_past",
        "product_company",
        "singular_plural",
        "synonym",
    ],
    "multiple_choice_qa": [
        "commonsense_qa",
        "math_qa",
        "science_qa",
        "history_qa",
        "geography_qa",
        "biology_qa",
        "chemistry_qa",
        "physics_qa",
        "literature_qa",
        "technology_qa",
        "sports_qa",
        "music_qa",
        "art_qa",
        "food_qa",
        "health_qa",
        "business_qa",
        "environment_qa",
        "psychology_qa",
        "language_qa",
        "animal_qa",
    ],
    "text_classification": [
        "sentiment_analysis",
        "topic_classification",
        "language_detection",
        "spam_detection",
        "ag_news",
        "genre_classification",
        "intent_classification",
        "emotion_detection",
        "difficulty_level",
        "urgency_classification",
        "formality_level",
        "age_group_target",
        "readability_level",
        "political_leaning",
        "safety_level",
        "bias_detection",
        "credibility_assessment",
        "content_rating",
        "complexity_level",
        "privacy_sensitivity",
    ],
    "extractive_tasks": [
        "adjective_vs_verb",
        "animal_vs_object",
        "choose_first_of_list",
        "choose_middle_of_list",
        "choose_last_of_list",
        "color_vs_animal",
        "concept_vs_object",
        "fruit_vs_animal",
        "object_vs_concept",
        "verb_vs_adjective",
        "living_vs_nonliving",
        "natural_vs_artificial",
        "singular_vs_plural_extractive",
        "concrete_vs_abstract",
        "positive_vs_negative",
        "past_vs_present",
        "question_vs_statement",
        "formal_vs_informal",
        "active_vs_passive",
        "literal_vs_figurative",
    ],
    "named_entity_recognition": [
        "ner_person",
        "ner_location",
        "ner_organization",
        "ner_date",
        "ner_number",
        "ner_product",
        "ner_currency",
        "ner_language",
        "ner_nationality",
        "ner_event",
        "ner_title",
        "ner_website",
        "ner_email",
        "ner_phone",
        "ner_address",
        "ner_time",
        "ner_percentage",
        "ner_age",
        "ner_duration",
        "ner_distance",
    ],
    "text_generation": [
        "complete_sentence",
        "continue_story",
        "writing_headlines",
        "question_generation",
        "dialogue_generation",
        "poetry_creation",
        "recipe_writing",
        "email_composition",
        "social_media_posts",
        "product_descriptions",
        "character_creation",
        "meeting_minutes",
        "technical_documentation",
        "creative_writing",
        "educational_content",
        "review_writing",
        "persuasive_writing",
        "instructional_content",
        "news_reporting",
        "scientific_writing",
    ],
}

FUNCTION_CATEGORIES = {
    "antonym": {
        "en": [
            "The opposite of 'happy' is",
            "What is the opposite of 'big'?",
            "Find an antonym for 'fast'",
            "Tell me a word that means the opposite of 'hot'",
            "What word would you use as the opposite of 'good'?"
        ],
        "de": [
            "Das Gegenteil von 'glücklich' ist",
            "Was ist das Gegenteil von 'groß'?",
            "Finde ein Antonym für 'schnell'",
            "Sag mir ein Wort, das das Gegenteil von 'heiß' bedeutet",
            "Welches Wort würdest du als Gegenteil von 'gut' verwenden?"
        ]
    },
    "capitalize": {
        "en": [
            "Converting 'hello' to its uppercase form gives",
            "What is 'world' written with an initial capital letter?",
            "The capitalized version of 'python' would be",
            "When capitalizing 'programming', it becomes",
            "Transform 'language' to start with a capital letter:"
        ],
        "de": [
            "Das Konvertieren von 'hallo' in seine Großbuchstabenform ergibt",
            "Was ist 'Welt' mit einem Anfangsbuchstaben geschrieben?",
            "Die großgeschriebene Version von 'Python' wäre",
            "Wenn man 'Programmierung' großschreibt, wird es",
            "Wandle 'Sprache' in Großbuchstaben um:"
        ]
    },
    "country_capital": {
        "en": [
            "The capital of France is",
            "What city serves as the capital of Japan?",
            "Name the capital city of Brazil",
            "Which city is the capital of Australia?",
            "If you were visiting Canada's capital, where would you be?"
        ],
        "de": [
            "Die Hauptstadt von Frankreich ist",
            "Welche Stadt dient als Hauptstadt von Japan?",
            "Nennen Sie die Hauptstadt von Brasilien",
            "Welche Stadt ist die Hauptstadt von Australien?",
            "Wenn Sie die Hauptstadt von Kanada besuchen würden, wo wären Sie?"
        ]
    },
    "country_currency": {
        "en": [
            "The official money used in Japan is",
            "What currency is spent in Australia?",
            "When traveling to Brazil, you would use which currency?",
            "The monetary unit of India is known as",
            "In Mexico, what currency do people use for transactions?"
        ],
        "de": [
            "Das offizielle Geld, das in Japan verwendet wird, ist",
            "Welche Währung wird in Australien ausgegeben?",
            "Wenn Sie nach Brasilien reisen, welche Währung würden Sie verwenden?",
            "Die Währungseinheit Indiens ist bekannt als",
            "In Mexiko, welche Währung verwenden die Menschen für Transaktionen?"
        ]
    },
    "translation_french": {
        "en": [
            "Translating 'hello' into French gives",
            "What would a French speaker say for 'world'?",
            "The equivalent of 'computer' in French is",
            "If you wanted to say 'book' in French, you would use",
            "In French, the word for 'friend' is translated as"
        ],
        "de": [
            "Die Übersetzung von 'hallo' ins Französische ergibt",
            "Was würde ein Französisch sprechender Mensch für 'Welt' sagen?",
            "Das Äquivalent von 'Computer' im Französischen ist",
            "Wenn du 'Buch' auf Französisch sagen wolltest, würdest du verwenden",
            "In Französisch ist das Wort für 'Freund' übersetzt als"
        ]
    },
    "translation_german": {
        "en": [
            "Translating 'hello' into German gives",
            "What would a German speaker say for 'world'?",
            "The equivalent of 'computer' in German is",
            "If you wanted to say 'book' in German, you would use",
            "In German, the word for 'friend' is translated as"
        ],
        "de": [
            "Die Übersetzung von 'hallo' ins Deutsche ergibt",
            "Was würde ein deutscher Sprecher für 'Welt' sagen?",
            "Das Äquivalent von 'Computer' auf Deutsch ist",
            "Wenn du 'Buch' auf Deutsch sagen wolltest, würdest du benutzen",
            "Auf Deutsch wird das Wort 'Freund' als übersetzt"
        ]
    },
    "translation_spanish": {
        "en": [
            "Translating 'hello' into Spanish gives",
            "What would a Spanish speaker say for 'world'?",
            "The equivalent of 'computer' in Spanish is",
            "If you wanted to say 'book' in Spanish, you would use",
            "In Spanish, the word for 'friend' is translated as"
        ],
        "de": [
            "Das Übersetzen von 'hallo' ins Spanische ergibt",
            "Was würde ein Spanischsprecher für 'Welt' sagen?",
            "Das Äquivalent von 'Computer' auf Spanisch ist",
            "Wenn du 'Buch' auf Spanisch sagen wolltest, würdest du verwenden",
            "Auf Spanisch wird das Wort für 'Freund' als übersetzt"
        ]
    },
    "landmark_country": {
        "en": [
            "The Eiffel Tower is located in",
            "What country is home to the Great Wall?",
            "The Taj Mahal can be found in which country?",
            "Which nation contains the Colosseum?",
            "The Great Pyramid is a monument in what country?"
        ],
        "de": [
            "Der Eiffelturm befindet sich in",
            "Welches Land ist Heimat der Chinesischen Mauer?",
            "In welchem Land kann das Taj Mahal gefunden werden?",
            "Welche Nation enthält das Kolosseum?",
            "Die Große Pyramide ist ein Denkmal in welchem Land?"
        ]
    },
    "lowercase": {
        "en": [
            "Converting 'HELLO' to small letters gives",
            "What is 'WORLD' written entirely in lowercase?",
            "The non-capitalized version of 'PYTHON' would be",
            "When changing 'PROGRAMMING' to lowercase, it becomes",
            "Transform 'LANGUAGE' to use only small letters:"
        ],
        "de": [
            "Das Umwandeln von 'HALLO' in Kleinbuchstaben ergibt",
            "Was ist 'WELT' vollständig in Kleinbuchstaben geschrieben?",
            "Die nicht-großgeschriebene Version von 'PYTHON' wäre",
            "Wenn man 'PROGRAMMING' in Kleinbuchstaben umwandelt, wird es",
            "Wandeln Sie 'LANGUAGE' so um, dass nur kleine Buchstaben verwendet werden:"
        ]
    },
    "national_parks": {
        "en": [
            "The country where Yellowstone National Park is located is",
            "Banff National Park is found in which nation?",
            "The Serengeti National Park is situated in",
            "Which country contains The Great Barrier Reef protected area?",
            "The nation where Kruger National Park is located is"
        ],
        "de": [
            "Das Land, in dem der Yellowstone-Nationalpark liegt, ist",
            "In welchem Land befindet sich der Banff-Nationalpark?",
            "Der Serengeti-Nationalpark befindet sich in",
            "Welches Land enthält das Schutzgebiet Großes Barriereriff?",
            "Das Land, in dem der Krüger-Nationalpark liegt, ist"
        ]
    },
    "next_item": {
        "en": [
            "The day that follows Monday is",
            "After January, the next month is",
            "The number that comes after 1 is",
            "Following spring, the next season is",
            "In the alphabet, the letter after A is"
        ],
        "de": [
            "Der Tag, der auf Montag folgt, ist",
            "Nach Januar ist der nächste Monat",
            "Die Zahl, die nach 1 kommt, ist",
            "Nach dem Frühling ist die nächste Jahreszeit",
            "Im Alphabet ist der Buchstabe nach A"
        ]
    },
    "previous_item": {
        "en": [
            "The day that comes before Tuesday is",
            "Prior to February, the previous month is",
            "The number that appears before 2 is",
            "Before summer, the preceding season is",
            "In the alphabet, the letter before B is"
        ],
        "de": [
            "Der Tag, der vor Dienstag kommt, ist",
            "Vor Februar ist der vorherige Monat",
            "Die Zahl, die vor 2 erscheint, ist",
            "Vor dem Sommer ist die vorherige Jahreszeit",
            "Im Alphabet ist der Buchstabe vor B"
        ]
    },
    "park_country": {
        "en": [
            "The nation containing Yellowstone National Park is",
            "Serengeti National Park is situated in which country?",
            "The geographical location of Banff National Park is",
            "Kruger National Park is found in which nation?",
            "The country where Plitvice Lakes National Park exists is"
        ],
        "de": [
            "Die Nation, die den Yellowstone-Nationalpark enthält, ist",
            "In welchem Land liegt der Serengeti-Nationalpark?",
            "Die geografische Lage des Banff-Nationalparks ist",
            "In welcher Nation befindet sich der Kruger-Nationalpark?",
            "Das Land, in dem der Plitvicer Seen-Nationalpark existiert, ist"
        ]
    },
    "person_instrument": {
        "en": [
            "The musical instrument played by a guitarist is",
            "What instrument is typically used by a pianist?",
            "A violinist performs using which musical tool?",
            "The instrument associated with a drummer is",
            "What device does a flutist use to make music?"
        ],
        "de": [
            "Das Musikinstrument, das von einem Gitarristen gespielt wird, ist",
            "Welches Instrument wird typischerweise von einem Pianisten verwendet?",
            "Welches musikalische Werkzeug verwendet ein Geiger?",
            "Das Instrument, das mit einem Schlagzeuger in Verbindung gebracht wird, ist",
            "Welches Gerät verwendet eine Flötistin, um Musik zu machen?"
        ]
    },
    "person_occupation": {
        "en": [
            "The workplace environment of a doctor is typically",
            "Where does a teacher generally perform their profession?",
            "The professional setting for a lawyer is usually",
            "A chef's workplace is commonly",
            "The occupational environment for a pilot is"
        ],
        "de": [
            "Die Arbeitsumgebung eines Arztes ist typischerweise",
            "Wo übt ein Lehrer ihren Beruf im Allgemeinen aus?",
            "Die berufliche Umgebung für einen Anwalt ist in der Regel",
            "Der Arbeitsplatz eines Kochs ist üblicherweise",
            "Die berufliche Umgebung für einen Piloten ist"
        ]
    },
    "person_sport": {
        "en": [
            "The athletic activity associated with a basketball player is",
            "What sport is practiced by a tennis player?",
            "The game that a golfer participates in is",
            "The athletic pursuit of a swimmer is",
            "What physical activity is performed by a runner?"
        ],
        "de": [
            "Die sportliche Aktivität, die mit einem Basketballspieler verbunden ist, ist",
            "Welche Sportart wird von einem Tennisspieler ausgeübt?",
            "Das Spiel, an dem ein Golfer teilnimmt, ist",
            "Die athletische Verfolgung eines Schwimmers ist",
            "Welche körperliche Aktivität wird von einem Läufer ausgeführt?"
        ]
    },
    "present_past": {
        "en": [
            "The past tense of 'run' is",
            "What is the past form of 'eat'?",
            "How would you say 'write' in past tense?",
            "Convert 'go' to its past tense form",
            "The word 'see' in past tense becomes"
        ],
        "de": [
            "Die Vergangenheitsform von 'laufen' ist",
            "Was ist die Vergangenheitsform von 'essen'?",
            "Wie würden Sie 'schreiben' in der Vergangenheitsform sagen?",
            "Wandeln Sie 'gehen' in seine Vergangenheitsform um",
            "Das Wort 'sehen' in der Vergangenheitsform wird"
        ]
    },
    "product_company": {
        "en": [
            "The corporation that manufactures the iPhone is",
            "Which company produces the Playstation?",
            "The Model S electric vehicle is manufactured by",
            "What organization develops the Windows operating system?",
            "The Galaxy smartphone line is created by which company?"
        ],
        "de": [
            "Die Gesellschaft, die das iPhone herstellt, ist",
            "Welches Unternehmen produziert die Playstation?",
            "Das Modell S Elektrofahrzeug wird von hergestellt",
            "Welche Organisation entwickelt das Windows-Betriebssystem?",
            "Die Galaxy-Smartphone-Reihe wird von welchem Unternehmen geschaffen?"
        ]
    },
    "singular_plural": {
        "en": [
            "When referring to multiple cat animals, we say",
            "The word for more than one dog is",
            "Converting 'book' to indicate many gives",
            "The plural form of the word 'child' is",
            "When referring to several mouse devices, we use"
        ],
        "de": [
            "Wenn wir auf mehrere Katzen tiere hinweisen, sagen wir",
            "Das Wort für mehr als einen Hund ist",
            "Die Umwandlung von 'Buch' um viele anzuzeigen gibt",
            "Die Mehrzahlform des Wortes 'Kind' ist",
            "Wenn wir auf mehrere Mäuse geräte hinweisen, verwenden wir"
        ]
    },
    "synonym": {
        "en": [
            "A synonym for 'happy' is",
            "What is another word for 'big'?",
            "Find a synonym for 'beautiful'",
            "Tell me a word that means the same as 'smart'",
            "What word would you use instead of 'fast'?"
        ],
        "de": [
            "Ein Synonym für 'glücklich' ist",
            "Was ist ein anderes Wort für 'groß'?",
            "Finde ein Synonym für 'schön'",
            "Sag mir ein Wort, das das Gleiche wie 'klug' bedeutet",
            "Welches Wort würdest du stattdessen für 'schnell' verwenden?"
        ]
    },
    "commonsense_qa": {
        "en": [
            "What is typically used to cut paper?",
            "Where is a salamander commonly found?",
            "Which appliance is the biggest electricity consumer in homes?",
            "What animal is known for barking?",
            "Where do people go when a movie is showing?"
        ],
        "de": [
            "Was wird typischerweise verwendet, um Papier zu schneiden?",
            "Wo findet man eine Salamander normalerweise?",
            "Welches Gerät ist der größte Stromverbraucher in den Haushalten?",
            "Welches Tier ist fürs Bellen bekannt?",
            "Wohin gehen Menschen, wenn ein Film gezeigt wird?"
        ]
    },
    "math_qa": {
        "en": [
            "What is the sum of 7 and 5?",
            "If x equals 3, what is the value of 2x + 4?",
            "What is the result when you find the square root of 25?",
            "What number is produced when 12 is multiplied by 3?",
            "What is the answer when 30 is divided by 6?"
        ],
        "de": [
            "Was ist die Summe von 7 und 5?",
            "Wenn x gleich 3 ist, was ist der Wert von 2x + 4?",
            "Was ist das Ergebnis, wenn du die Quadratwurzel aus 25 findest?",
            "Welche Zahl entsteht, wenn 12 mit 3 multipliziert wird?",
            "Welche Antwort ergibt sich, wenn 30 durch 6 geteilt wird?"
        ]
    },
    "science_qa": {
        "en": [
            "What is H2O more commonly known as?",
            "Which planet is positioned closest to the Sun?",
            "What process is used by plants to create their food?",
            "What natural substance is considered the hardest on Earth?",
            "What force is responsible for pulling objects toward Earth?"
        ],
        "de": [
            "Als was ist H2O allgemeiner bekannt?",
            "Welcher Planet befindet sich am nächsten zur Sonne?",
            "Welcher Prozess wird von Pflanzen verwendet, um ihre Nahrung zu erzeugen?",
            "Welches natürliche Substanz gilt als härtestes auf der Erde?",
            "Welche Kraft ist verantwortlich für das Ziehen von Objekten zur Erde?"
        ]
    },
    "history_qa": {
        "en": [
            "Who is recognized as the first president of the United States?",
            "In what year is World War II recorded to have ended?",
            "Who is credited with painting the Mona Lisa?",
            "Which ancient civilization is known for building the pyramids?",
            "Who is the author of the Declaration of Independence?"
        ],
        "de": [
            "Wer wird als erster Präsident der Vereinigten Staaten anerkannt?",
            "In welchem Jahr wird der Zweite Weltkrieg als beendet eingetragen?",
            "Wer wird mit dem Malen der Mona Lisa genannt?",
            "Welche antike Zivilisation ist bekannt für das Bauen der Pyramiden?",
            "Wer ist der Autor der Unabhängigkeitserklärung?"
        ]
    },
    "geography_qa": {
        "en": [
            "Which body of water is Earth's largest ocean?",
            "What city is the capital of Canada?",
            "Which country has the world's largest population?",
            "What is considered the longest river in the world?",
            "On which continent is the Sahara Desert located?"
        ],
        "de": [
            "Welcher Gewässerkörper ist der größte Ozean der Erde?",
            "Welche Stadt ist die Hauptstadt von Kanada?",
            "Welches Land hat die größte Bevölkerung der Welt?",
            "Welcher Fluss gilt als der längste Fluss der Welt?",
            "Auf welchem Kontinent liegt die Sahara-Wüste?"
        ]
    },
    "biology_qa": {
        "en": [
            "What is the most basic unit of life?",
            "Which organ is responsible for pumping blood?",
            "What process allows plants to convert sunlight into energy?",
            "Which gas do animals breathe out?",
            "What is the genetic material found in cells?"
        ],
        "de": [
            "Was ist die grundlegendste Lebensbaeinheit?",
            "Welches Organ ist verantwortlich für das Pumpen von Blut?",
            "Welcher Prozess ermöglicht es Pflanzen, Sonnenlicht in Energie umzuwandeln?",
            "Welches Gas atmen Tiere aus?",
            "Was ist das genetische Material, das in Zellen gefunden wird?"
        ]
    },
    "chemistry_qa": {
        "en": [
            "What is the chemical symbol for gold?",
            "Which element is most abundant in Earth's atmosphere?",
            "What is formed when acid and base react?",
            "Which state of matter has particles close together but movable?",
            "What type of bond forms between metals and non-metals?"
        ],
        "de": [
            "Was ist das chemische Symbol für Gold?",
            "Welches Element ist am häufigsten in der Erdatmosphäre vorhanden?",
            "Was entsteht, wenn Säure und Base reagieren?",
            "Welcher Aggregatzustand hat Teilchen, die dicht beieinander liegen, aber beweglich sind?",
            "Welche Art von Bindung bildet sich zwischen Metallen und Nichtmetallen?"
        ]
    },
    "physics_qa": {
        "en": [
            "What is the unit of measurement for electric current?",
            "Which law states that energy cannot be created or destroyed?",
            "What is the speed of light in a vacuum?",
            "Which force keeps planets in orbit around the sun?",
            "What happens to wavelength when frequency increases?"
        ],
        "de": [
            "Was ist die Einheit der Messung für elektrischen Strom?",
            "Welches Gesetz besagt, dass Energie nicht geschaffen oder vernichtet werden kann?",
            "Was ist die Lichtgeschwindigkeit im Vakuum?",
            "Welche Kraft hält die Planeten auf ihrer Bahn um die Sonne?",
            "Was passiert mit der Wellenlänge, wenn die Frequenz zunimmt?"
        ]
    },
    "literature_qa": {
        "en": [
            "Who is the author of 'Pride and Prejudice'?",
            "What is the first book in the Harry Potter series?",
            "Which play features the character of Hamlet?",
            "Who wrote 'To Kill a Mockingbird'?",
            "What is the setting of '1984' by George Orwell?"
        ],
        "de": [
            "Wer ist der Autor von 'Stolz und Vorurteil'?",
            "Was ist das erste Buch in der Harry Potter Reihe?",
            "Welches Theaterstück enthält die Figur des Hamlet?",
            "Wer schrieb 'Töte einen Spottvogel'?",
            "Was ist die Handlungsszene von '1984' von George Orwell?"
        ]
    },
    "technology_qa": {
        "en": [
            "What does CPU stand for in computers?",
            "Which company developed the Android operating system?",
            "What is the primary function of RAM in a computer?",
            "Which programming language is known for web development?",
            "What does WWW stand for?"
        ],
        "de": [
            "Was bedeutet CPU in Computern?",
            "Welches Unternehmen hat das Android-Betriebssystem entwickelt?",
            "Was ist die Hauptfunktion von RAM in einem Computer?",
            "Welche Programmiersprache ist für die Webentwicklung bekannt?",
            "Was bedeutet WWW?"
        ]
    },
    "sports_qa": {
        "en": [
            "Which sport is played at Wimbledon?",
            "How many players are on a basketball team on the court?",
            "What is the maximum score possible in ten-pin bowling?",
            "Which country hosted the 2016 Summer Olympics?",
            "What is the duration of a soccer match?"
        ],
        "de": [
            "Welcher Sport wird bei Wimbledon gespielt?",
            "Wie viele Spieler sind bei einem Basketballspiel auf dem Feld?",
            "Was ist die höchstmögliche Punktzahl im Zehn-Pin-Bowling?",
            "Welches Land veranstaltete die Olympischen Sommerspiele 2016?",
            "Wie lange dauert ein Fußballspiel?"
        ]
    },
    "music_qa": {
        "en": [
            "Which instrument has 88 keys?",
            "What is the highest vocal range for a female singer?",
            "How many strings does a standard guitar have?",
            "Which composer wrote 'The Four Seasons'?",
            "What does 'forte' mean in musical terms?"
        ],
        "de": [
            "Welches Instrument hat 88 Tasten?",
            "Was ist der höchste Vokalbereich für eine weibliche Sängerin?",
            "Wie viele Saiten hat eine Standardgitarre?",
            "Welcher Komponist schrieb 'Die Vier Jahreszeiten'?",
            "Was bedeutet 'forte' in musikalischer Hinsicht?"
        ]
    },
    "art_qa": {
        "en": [
            "Who painted 'The Starry Night'?",
            "What is the primary color that cannot be created by mixing?",
            "Which art movement did Pablo Picasso help establish?",
            "What material is traditionally used for sculpting statues?",
            "Which museum houses the Mona Lisa?"
        ],
        "de": [
            "Wer malte 'Die Sternennacht'?",
            "Welche Grundfarbe kann nicht durch Mischen geschaffen werden?",
            "Welche Kunstbewegung half Pablo Picasso etablieren?",
            "Welches Material wird traditionell für das Formen von Statuen verwendet?",
            "Welches Museum beherbergt die Mona Lisa?"
        ]
    },
    "food_qa": {
        "en": [
            "What is the main ingredient in hummus?",
            "Which spice is derived from the Crocus flower?",
            "What type of pastry is used to make profiteroles?",
            "Which country is credited with inventing pizza?",
            "What is the most consumed beverage in the world after water?"
        ],
        "de": [
            "Welches ist das Hauptzutat in Hummus?",
            "Welcher Gewürz wird aus der Safranblume abgeleitet?",
            "Welche Art von Gebäck wird verwendet, um Profiteroles herzustellen?",
            "Welches Land wird mit der Erfindung der Pizza geehrt?",
            "Welches Getränk wird nach Wasser am häufigsten weltweit konsumiert?"
        ]
    },
    "health_qa": {
        "en": [
            "What is the normal human body temperature in Celsius?",
            "Which vitamin is primarily obtained from sunlight?",
            "What is the main function of red blood cells?",
            "Which part of the body is affected by glaucoma?",
            "What is the recommended daily intake of water?"
        ],
        "de": [
            "Welche ist die normale Körpertemperatur des Menschen in Celsius?",
            "Welches Vitamin wird hauptsächlich durch Sonnenlicht aufgenommen?",
            "Welche ist die Hauptfunktion der roten Blutkörperchen?",
            "Welcher Teil des Körpers wird von der Glaukom betroffen?",
            "Was ist die empfohlene tägliche Aufnahme von Wasser?"
        ]
    },
    "business_qa": {
        "en": [
            "What does ROI stand for in business?",
            "Who is the CEO of Tesla, Inc.?",
            "What is a bull market?",
            "Which stock exchange is located in New York City?",
            "What is a 'unicorn' in the business world?"
        ],
        "de": [
            "Was bedeutet ROI im Geschäft?",
            "Wer ist der CEO von Tesla, Inc.?",
            "Was ist ein Bullenmarkt?",
            "An welcher Börse befindet sich in New York City?",
            "Was ist ein 'Einhorn' in der Geschäftswelt?"
        ]
    },
    "environment_qa": {
        "en": [
            "What is the main cause of global warming?",
            "Which gas makes up the majority of Earth's atmosphere?",
            "What is the process of converting waste into reusable material?",
            "Which layer of the atmosphere protects Earth from UV radiation?",
            "What is the term for the variety of life on Earth?"
        ],
        "de": [
            "Was ist die Hauptursache der globalen Erwärmung?",
            "Welches Gas macht den Großteil der Erdatmosphäre aus?",
            "Was ist der Prozess der Umwandlung von Abfällen in wiederverwendbares Material?",
            "Welche Schicht der Atmosphäre schützt die Erde vor UV-Strahlung?",
            "Was ist der Begriff für die Vielfalt des Lebens auf der Erde?"
        ]
    },
    "psychology_qa": {
        "en": [
            "Who is considered the father of psychoanalysis?",
            "What is the 'fight or flight' response?",
            "What is cognitive dissonance?",
            "Which part of the brain is responsible for decision-making?",
            "What is the term for a persistent fear of an object or situation?"
        ],
        "de": [
            "Wer gilt als der Vater der Psychoanalyse?",
            "Was ist die 'Kampf-oder-Flucht'-Reaktion?",
            "Was ist kognitive Dissonanz?",
            "Welcher Teil des Gehirns ist für Entscheidungsfindung verantwortlich?",
            "Was ist der Begriff für eine dauerhafte Angst vor einem Objekt oder einer Situation?"
        ]
    },
    "language_qa": {
        "en": [
            "What is a verb?",
            "How many letters are in the English alphabet?",
            "What is the term for a word that sounds the same as another but has a different meaning?",
            "Which language is the most spoken in the world by number of native speakers?",
            "What is a palindrome?"
        ],
        "de": [
            "Was ist ein Verb?",
            "Wie viele Buchstaben gibt es im englischen Alphabet?",
            "Was ist der Begriff für ein Wort, das genauso klingt wie ein anderes, aber eine andere Bedeutung hat?",
            "Welche Sprache wird am häufigsten auf der Welt von Muttersprachlern gesprochen?",
            "Was ist ein Palindrom?"
        ]
    },
    "animal_qa": {
        "en": [
            "What is the largest mammal in the world?",
            "Which animal is known as the 'king of the jungle'?",
            "How many legs does a spider have?",
            "What is a group of lions called?",
            "Which bird is a symbol of peace?"
        ],
        "de": [
            "Was ist das größte Säugetier der Welt?",
            "Welches Tier wird als 'König des Dschungels' bekannt?",
            "Wie viele Beine hat eine Spinne?",
            "Was wird eine Gruppe von Löwen genannt?",
            "Welcher Vogel ist ein Symbol für Frieden?"
        ]
    },
    "sentiment_analysis": {
        "en": [
            "Classify the sentiment of the following sentence: 'I absolutely loved the movie, it was fantastic!'",
            "Is the sentiment of 'The service was terrible and the food was cold' positive or negative?",
            "Determine the sentiment of the review: 'An average experience, nothing special.'",
            "What is the sentiment of the text: 'I am so excited for the concert tonight!'?",
            "Classify the following statement as positive, negative, or neutral: 'The package arrived on time.'"
        ],
        "de": [
            "Klassifiziere die Stimmung des folgenden Satzes: 'Ich habe den Film absolut geliebt, er war fantastisch!'",
            "Ist die Stimmung von 'Der Service war schrecklich und das Essen war kalt' positiv oder negativ?",
            "Bestimme die Stimmung der Bewertung: 'Eine durchschnittliche Erfahrung, nichts Besonderes.'",
            "Was ist die Stimmung des Textes: 'Ich bin so aufgeregt auf das Konzert heute Abend!'?",
            "Klassifizieren Sie die folgende Aussage als positiv, negativ oder neutral: 'Das Paket ist pünktlich angekommen.'"
        ]
    },
    "topic_classification": {
        "en": [
            "What is the topic of the sentence: 'The stock market reached a new high today as tech companies soared.'?",
            "Classify the topic of the following text: 'The new space telescope has discovered a planet with two suns.'",
            "Determine the subject of the article starting with: 'The local government announced new policies on recycling.'",
            "What is the main theme of a text about 'the causes and effects of climate change'?",
            "Categorize the following sentence into a topic: 'The home team won the championship in a thrilling final match.'"
        ],
        "de": [
            "Welches ist das Thema des Satzes: 'Der Aktienmarkt erreichte heute einen neuen Höchststand, da Technologieunternehmen in die Höhe schossen.'?",
            "Klassifizieren Sie das Thema des folgenden Textes: 'Das neue Weltraumteleskop hat einen Planeten mit zwei Sonnen entdeckt.'",
            "Bestimmen Sie das Thema des Artikels, der mit folgendem Satz beginnt: 'Die lokale Regierung kündigte neue Recyclingrichtlinien an.'",
            "Welches ist das Hauptthema eines Textes über 'die Ursachen und Auswirkungen des Klimawandels'?",
            "Kategorisieren Sie den folgenden Satz in ein Thema: 'Das Heimteam gewann die Meisterschaft in einem spannenden Finale.'"
        ]
    },
    "language_detection": {
        "en": [
            "Identify the language of the following text: 'Bonjour, comment ça va?'",
            "What language is this sentence written in: 'La vida es bella.'?",
            "Detect the language of: 'Guten Tag, wie geht es Ihnen?'",
            "Which language is being used in the phrase: 'Eu não falo português.'?",
            "Identify the language of the text: 'Привет, как дела?'"
        ],
        "de": [
            "Erkennen Sie die Sprache des folgenden Textes: 'Bonjour, comment ça va?'",
            "Welche Sprache ist dieser Satz geschrieben in: 'La vida es bella.'?",
            "Erkennen Sie die Sprache von: 'Guten Tag, wie geht es Ihnen?'",
            "Welche Sprache wird in der Phrase verwendet: 'Eu não falo português.'?",
            "Erkennen Sie die Sprache des Textes: 'Привет, как дела?'"
        ]
    },
    "spam_detection": {
        "en": [
            "Is the following email likely spam or not? 'Congratulations, you've won a million dollars! Click here to claim your prize.'",
            "Classify this message: 'Your account has been compromised. Please verify your details immediately.'",
            "Determine if this is a spam message: 'Exclusive offer just for you! Get 50% off now.'",
            "Is the following text spam? 'Hi, it's me. I'm running late for the meeting.'",
            "Classify the email with the subject 'You have a new voice message' as spam or not spam."
        ],
        "de": [
            "Ist die folgende E-Mail wahrscheinlich Spam oder nicht? 'Herzlichen Glückwunsch, Sie haben einen Millionen Dollar gewonnen! Klicken Sie hier, um Ihren Preis einzulösen.'",
            "Klassifizieren Sie diese Nachricht: 'Ihr Konto wurde kompromittiert. Bitte überprüfen Sie Ihre Daten sofort.'",
            "Bestimmen Sie, ob dies eine Spam-Nachricht ist: 'Exklusives Angebot nur für Sie! Holen Sie sich jetzt 50% Rabatt.'",
            "Ist der folgende Text Spam? 'Hallo, ich bin es. Ich komme zu spät zur Besprechung.'",
            "Klassifizieren Sie die E-Mail mit dem Betreff 'Sie haben eine neue Sprachnachricht' als Spam oder kein Spam."
        ]
    },
    "ag_news": {
        "en": [
            "Categorize the following news headline: 'Scientists Discover New Species of Deep-Sea Fish.'",
            "What is the category of the news: 'The Federal Reserve announced an interest rate hike.'?",
            "Classify the headline: 'The Lakers win the NBA championship.'",
            "Determine the topic of the news: 'A new AI model has been released that can generate realistic images from text.'",
            "What category does this headline belong to? 'The famous actor announced their retirement from acting.'"
        ],
        "de": [
            "Kategorisieren Sie die folgende Nachrichtenüberschrift: 'Wissenschaftler entdecken neue Art von Tiefseefischen.'",
            "Welche Kategorie hat die Nachricht: 'Die Federal Reserve kündigte eine Zinserhöhung an.'?",
            "Klassifiziere die Schlagzeile: 'Die Lakers gewinnen die NBA-Meisterschaft.'",
            "Bestimme das Thema der Nachricht: 'Ein neues KI-Modell wurde veröffentlicht, das realistische Bilder aus Text generieren kann.'",
            "Welcher Kategorie gehört diese Schlagzeile an? 'Der berühmte Schauspieler kündigte ihren Rückzug aus dem Schauspiel an.'"
        ]
    },
    "genre_classification": {
        "en": [
            "What is the genre of a story that begins: 'In a galaxy far, far away...'?",
            "Classify the genre of a book with the description: 'A detective tries to solve a murder in a small town.'",
            "Determine the genre of a movie about a young wizard who goes to a magical school.",
            "What is the genre of the following plot: 'A group of friends go on a quest to destroy a powerful, evil artifact.'?",
            "Categorize a story about a spaceship crew exploring a new planet."
        ],
        "de": [
            "Welches ist der Genre einer Geschichte, die beginnt: 'In einer Galaxie weit, weit weg...'?",
            "Klassifiziere das Genre eines Buches mit der Beschreibung: 'Ein Detektiv versucht, einen Mord in einer kleinen Stadt zu lösen.'",
            "Bestimme das Genre eines Films über einen jungen Zauberer, der auf eine magische Schule geht.",
            "Welcher ist der Genre des folgenden Plots: 'Eine Gruppe von Freunden geht auf eine Quest, um ein mächtiges, böses Artefakt zu zerstören.'?",
            "Kategorisiere eine Geschichte über eine Raumschiffbesatzung, die einen neuen Planeten erkundet."
        ]
    },
    "intent_classification": {
        "en": [
            "What is the user's intent in the following query: 'Find me a good Italian restaurant near me.'?",
            "Classify the intent of the question: 'What is the weather like today?'",
            "Determine the user's goal in the command: 'Set an alarm for 7 AM tomorrow.'",
            "What is the intent behind the query: 'How do I change my password?'?",
            "Categorize the user's intent: 'Play some relaxing music.'"
        ],
        "de": [
            "Welche ist die Absicht des Benutzers in der folgenden Anfrage: 'Finde mir ein gutes italienisches Restaurant in meiner Nähe.'?",
            "Klassifiziere die Absicht der Frage: 'Wie ist das Wetter heute?'",
            "Bestimme das Ziel des Benutzers im Befehl: 'Stelle einen Wecker für 7 Uhr morgens morgen ein.'",
            "Welche ist die Absicht hinter der Anfrage: 'Wie ändere ich mein Passwort?'?",
            "Kategorisiere die Absicht des Benutzers: 'Spiele einige beruhigende Musik.'"
        ]
    },
    "emotion_detection": {
        "en": [
            "What emotion is expressed in the sentence: 'I am so incredibly happy and grateful for this award!'?",
            "Identify the emotion in the text: 'I can't believe he would betray me like that.'",
            "Detect the emotion in: 'I'm terrified of what might happen next.'",
            "What is the primary emotion in the statement: 'I'm just so tired and overwhelmed with all this work.'?",
            "Classify the emotion of the sentence: 'Wow, this is a wonderful surprise!'"
        ],
        "de": [
            "Welche Emotion wird in dem Satz ausgedrückt: 'Ich bin so unglaublich glücklich und dankbar für diesen Preis!'?",
            "Identifiziere die Emotion im Text: 'Ich kann nicht glauben, dass er mich so verraten würde.'",
            "Erkenne die Emotion in: 'Ich habe schreckliche Angst vor dem, was als Nächstes passieren könnte.'",
            "Welche ist die primäre Emotion in der Aussage: 'Ich bin einfach so müde und überwältigt von all dieser Arbeit.'?",
            "Klassifiziere die Emotion des Satzes: 'Wow, das ist eine wunderbare Überraschung!'"
        ]
    },
    "difficulty_level": {
        "en": [
            "Rate the difficulty of the following question: 'What is 2+2?'",
            "Is the text 'Quantum mechanics is the study of matter and energy at the most fundamental level' easy, medium, or hard to understand?",
            "Determine the difficulty level of the task: 'Write a sonnet in the style of Shakespeare.'",
            "Classify the complexity of the question: 'Explain the theory of general relativity.'",
            "Is the following instruction easy or hard? 'Assemble the provided flat-pack furniture using these diagrams.'"
        ],
        "de": [
            "Bewerte die Schwierigkeit der folgenden Frage: 'Was ist 2+2?'",
            "Ist der Text 'Quantenmechanik ist das Studium von Materie und Energie auf der grundlegendsten Ebene' leicht, mittel oder schwer zu verstehen?",
            "Bestimme die Schwierigkeitsstufe der Aufgabe: 'Schreibe ein Sonett im Stil von Shakespeare.'",
            "Klassifiziere die Komplexität der Frage: 'Erkläre die Theorie der allgemeinen Relativität.'",
            "Ist die folgende Anweisung leicht oder schwer? 'Bau das bereitgestellte Flachpaket-Möbel mit diesen Diagrammen zusammen.'"
        ]
    },
    "urgency_classification": {
        "en": [
            "Is the following message urgent? 'The building is on fire, evacuate immediately!'",
            "Classify the urgency of the request: 'Can you please send me the report by the end of the day?'",
            "Determine if this situation is urgent: 'My computer has crashed and I have a deadline in an hour.'",
            "Is this message high-priority? 'Reminder: Team meeting tomorrow at 10 AM.'",
            "Classify the urgency of: 'I've run out of milk, can you pick some up on your way home?'"
        ],
        "de": [
            "Ist die folgende Nachricht dringend? 'Das Gebäude ist in Flammen, evakuiere sofort!'",
            "Ermittle die Dringlichkeit der Anfrage: 'Kannst du mir bitte den Bericht bis zum Ende des Tages schicken?'",
            "Bestimme, ob diese Situation dringend ist: 'Mein Computer ist abgestürzt und ich habe eine Frist in einer Stunde.'",
            "Ist diese Nachricht hochprioritär? 'Erinnerung: Teamtreffen morgen um 10 Uhr.'",
            "Ermittle die Dringlichkeit von: 'Ich habe keine Milch mehr, kannst du auf dem Heimweg welche besorgen?'"
        ]
    },
    "formality_level": {
        "en": [
            "Is the following sentence formal or informal? 'Hey, what's up?'",
            "Classify the formality of the statement: 'To whom it may concern, I am writing to apply for the position advertised.'",
            "Determine the formality level of: 'Dude, that's awesome!'",
            "Is the sentence 'The meeting is scheduled to commence at 3:00 PM' formal or informal?",
            "Classify the formality of 'Let's grab a bite to eat later.'"
        ],
        "de": [
            "Ist der folgende Satz formal oder informell? 'Hey, was ist los?'",
            "Ermittle die Formalität der Aussage: 'An wen es auch betrifft, ich schreibe, um auf die beworbene Stelle zu bewerben.'",
            "Bestimme die Formalitätsstufe von: 'Mann, das ist super!'",
            "Ist der Satz 'Die Besprechung ist für 15:00 Uhr angesetzt' formal oder informell?",
            "Ermittle die Formalität von 'Lass uns später was essen gehen.'"
        ]
    },
    "age_group_target": {
        "en": [
            "Is the book 'The Very Hungry Caterpillar' intended for children or adults?",
            "What is the target age group for the movie 'Toy Story'?",
            "Classify the content 'A research paper on quantum physics' by its target audience's age.",
            "Determine the intended age group for a video game with a rating of 'Mature 17+.'",
            "Is the TV show 'Sesame Street' for kids or adults?"
        ],
        "de": [
            "Ist das Buch 'Der sehr hungrige Raupe' für Kinder oder Erwachsene gedacht?",
            "Welche Altersgruppe ist das Ziel für den Film 'Toy Story'?",
            "Ermittle den Inhalt 'Eine Forschungsarbeit über Quantenphysik' nach dem Zielpublikumsalter.",
            "Bestimme die vorgesehene Altersgruppe für ein Videospiel mit der Einstufung 'Erwachsene 17+'.",
            "Ist die Fernsehserie 'Sesamstraße' für Kinder oder Erwachsene?"
        ]
    },
    "readability_level": {
        "en": [
            "What is the readability level of the sentence 'The cat sat on the mat.'?",
            "Assess the readability of the text: 'The epistemological ramifications of post-structuralist discourse are profound.'",
            "Is the following text easy or difficult to read? 'See Spot run. Run, Spot, run.'",
            "Determine the reading level required for a legal contract.",
            "Classify the readability of a typical academic journal article."
        ],
        "de": [
            "Welches ist das Leseverständnisniveau des Satzes 'Die Katze saß auf der Matte.'?",
            "Beurteile das Leseverständnis des Textes: 'Die epistemologischen Auswirkungen des poststrukturalistischen Diskurses sind tiefgreifend.'",
            "Ist der folgende Text leicht oder schwierig zu lesen? 'Sieh Spot laufen. Lauf, Spot, lauf.'",
            "Bestimme das Leseniveau, das für einen Vertrag erforderlich ist.",
            "Ermittle die Lesbarkeit eines typischen wissenschaftlichen Zeitschriftenartikels."
        ]
    },
    "political_leaning": {
        "en": [
            "What is the likely political leaning of an article that strongly advocates for lower taxes and less government regulation?",
            "Classify the political stance of a speech that emphasizes social welfare programs and environmental protection.",
            "Determine the political orientation of a news source that is described as 'right-leaning.'",
            "What political ideology is most associated with the phrase 'Workers of the world, unite!'?",
            "Is an argument for free-market capitalism typically considered left-wing or right-wing?"
        ],
        "de": [
            "Welche politische Ausrichtung ist ein Artikel wahrscheinlich, der sich stark für niedrigere Steuern und weniger staatliche Regulierung ausspricht?",
            "Ermittle die politische Position einer Rede, die soziale Wohlfahrtsprogramme und Umweltschutz betont.",
            "Bestimme die politische Ausrichtung einer Nachrichtenquelle, die als 'rechtslastig' beschrieben wird.",
            "Welche politische Ideologie ist am stärksten mit dem Satz 'Arbeiter der Welt, vereinigt Euch!' verbunden?",
            "Wird ein Argument für freien Marktkapitalismus typischerweise als links- oder rechtsorientiert betrachtet?"
        ]
    },
    "safety_level": {
        "en": [
            "Is the following content safe for all audiences? 'A guide to baking a chocolate cake.'",
            "Classify the safety of the text: 'Instructions on how to build a homemade explosive device.'",
            "Determine if the website 'www.cutepuppies.com' is likely to be safe for work.",
            "Is the following query safe? 'How to tie a tie.'",
            "Classify the content 'A graphic depiction of violence' in terms of safety."
        ],
        "de": [
            "Ist der folgende Inhalt für alle Zielgruppen geeignet? 'Eine Anleitung zum Backen einer Schokoladenkuchen.'",
            "Klassifizieren Sie die Sicherheit des Textes: 'Anweisungen zum Bau einer selbstgemachten Sprengvorrichtung.'",
            "Bestimmen Sie, ob die Website 'www.cutepuppies.com' wahrscheinlich arbeitsplatzgeeignet ist.",
            "Ist die folgende Anfrage sicher? 'Wie man eine Krawatte bindet.'",
            "Klassifizieren Sie den Inhalt 'Eine grafische Darstellung von Gewalt' hinsichtlich der Sicherheit."
        ]
    },
    "bias_detection": {
        "en": [
            "Does the sentence 'All politicians are corrupt' show bias?",
            "Identify if there is a bias in the statement: 'This new product is the best thing since sliced bread, everyone should buy it.'",
            "Is the following text biased? 'The home team was robbed by a terrible refereeing decision.'",
            "Detect bias in the sentence: 'People from that country are always so lazy.'",
            "Does the statement 'According to a study by...' contain less bias than 'In my opinion...'?"
        ],
        "de": [
            "Zeigt der Satz 'Alle Politiker sind korrupt' Vorurteile?",
            "Erkennen Sie, ob in der Aussage ein Bias vorhanden ist: 'Dieses neue Produkt ist das Beste seit geschnittenem Brot, jeder sollte es kaufen.'",
            "Ist der folgende Text verzerrt? 'Das Heimteam wurde durch eine schreckliche Schiedsrichterentscheidung bestohlen.'",
            "Erkennen Sie Bias in dem Satz: 'Menschen aus diesem Land sind immer so faul.'",
            "Enthält die Aussage 'Nach einer Studie von...' weniger Bias als 'Meiner Meinung nach...'?"
        ]
    },
    "credibility_assessment": {
        "en": [
            "Is a peer-reviewed scientific journal a credible source of information?",
            "Assess the credibility of a random blog post written by an anonymous author.",
            "Is a statement from a government health organization likely to be credible?",
            "How credible is a rumor you heard from a friend of a friend?",
            "Evaluate the credibility of a news article that cites multiple, verifiable sources."
        ],
        "de": [
            "Ist eine peer-begutachtete wissenschaftliche Zeitschrift eine vertrauenswürdige Informationsquelle?",
            "Bewerten Sie die Glaubwürdigkeit eines zufälligen Blog-Beitrags, der von einem anonymen Autor geschrieben wurde.",
            "Ist eine Aussage von einer Regierungs-Gesundheitsorganisation wahrscheinlich glaubwürdig?",
            "Wie glaubwürdig ist ein Gerücht, das du von einem Freund eines Freundes gehört hast?",
            "Bewerte die Glaubwürdigkeit eines Nachrichtenartikels, der mehrere überprüfbare Quellen nennt."
        ]
    },
    "content_rating": {
        "en": [
            "What would be an appropriate content rating for a film that contains strong language and violence?",
            "Rate the movie 'Finding Nemo' for its suitability for young children.",
            "What content rating would you assign to a video game that features realistic gambling?",
            "Assign a content rating to a documentary about the history of art.",
            "What is the likely rating of a TV show described as a 'family-friendly sitcom'?"
        ],
        "de": [
            "Welche Inhaltsbewertung wäre für einen Film angemessen, der starke Sprache und Gewalt enthält?",
            "Bewerte den Film 'Findet Nemo' für seine Eignung für junge Kinder.",
            "Welche Inhaltsbewertung würdest du einem Videospiel zuweisen, das realistisches Glücksspiel zeigt?",
            "Weisen Sie einer Dokumentation über die Geschichte der Kunst eine Inhaltsbewertung zu.",
            "Was ist die wahrscheinliche Bewertung einer Fernsehshow, die als 'familienfreundliche Sitcom' beschrieben wird?"
        ]
    },
    "complexity_level": {
        "en": [
            "What is the complexity of the task 'counting to ten'?",
            "Assess the complexity of 'solving a Rubik's cube'.",
            "Is the process of 'breathing' simple or complex?",
            "Determine the complexity of 'building a car from scratch'.",
            "Classify the complexity of the game of 'tic-tac-toe'."
        ],
        "de": [
            "Was ist die Komplexität der Aufgabe 'bis zehn zählen'?",
            "Bewerten Sie die Komplexität von 'einem Zauberwürfel zu lösen'.",
            "Ist der Prozess des 'Atmens' einfach oder komplex?",
            "Bestimmen Sie die Komplexität von 'einem Auto von Grund auf zu bauen'.",
            "Klassifiziere die Komplexität des Spiels 'Kreuz-Null'."
        ]
    },
    "privacy_sensitivity": {
        "en": [
            "Is the information 'What is your favorite color?' considered private?",
            "How sensitive is the data 'your social security number'?",
            "Assess the privacy sensitivity of 'your home address'.",
            "Is 'your opinion on the weather' private information?",
            "Determine the sensitivity of 'your medical records'."
        ],
        "de": [
            "Ist die Information 'Was ist deine Lieblingsfarbe?' als privat zu betrachten?",
            "Wie sensibel sind die Daten 'deine Sozialversicherungsnummer'?",
            "Beurteile die Datenschutzsensitivität von 'deiner Heimatadresse'.",
            "Ist 'deine Meinung zum Wetter' private Information?",
            "Bestimme die Sensitivität von 'deinen Krankenakten'."
        ]
    },
    "adjective_vs_verb": {
        "en": [
            "Is 'beautiful' an adjective or a verb?",
            "Identify if 'run' is an adjective or a verb.",
            "Choose the adjective from the following pair: 'quick', 'jump'.",
            "Which of these is a verb: 'sleep' or 'sleepy'?",
            "Determine if 'happy' describes an action or a quality."
        ],
        "de": [
            "Ist 'schön' ein Adjektiv oder ein Verb?",
            "Erkennen Sie, ob 'laufen' ein Adjektiv oder ein Verb ist.",
            "Wählen Sie das Adjektiv aus dem folgenden Paar: 'schnell', 'springen'.",
            "Welches von diesen ist ein Verb: 'schlafen' oder 'schlaftrunken'?",
            "Bestimmen Sie, ob 'glücklich' eine Handlung oder eine Eigenschaft beschreibt."
        ]
    },
    "animal_vs_object": {
        "en": [
            "Is a 'dog' an animal or an object?",
            "Classify 'chair' as either an animal or an object.",
            "Which of the following is an animal: 'cat' or 'car'?",
            "Determine if a 'table' is a living creature or an inanimate object.",
            "Choose the object from the pair: 'bird', 'book'."
        ],
        "de": [
            "Ist ein 'Hund' ein Tier oder ein Objekt?",
            "Klassifiziere 'Stuhl' als entweder ein Tier oder ein Objekt.",
            "Welches der folgenden ist ein Tier: 'Katze' oder 'Auto'?",
            "Bestimme, ob ein 'Tisch' ein lebendes Wesen oder ein unbelebtes Objekt ist.",
            "Wähle das Objekt aus dem Paar: 'Vogel', 'Buch'."
        ]
    },
    "choose_first_of_list": {
        "en": [
            "From the list 'apple, banana, cherry', what is the first item?",
            "What is the first element in the sequence '1, 2, 3'?",
            "Given 'red, green, blue', which color is listed first?",
            "Identify the first word in the sentence: 'The quick brown fox.'",
            "From the options 'A, B, C', select the first letter."
        ],
        "de": [
            "Aus der Liste 'Apfel, Banane, Kirsche', was ist das erste Element?",
            "Was ist das erste Element in der Sequenz '1, 2, 3'?",
            "Gegeben 'rot, grün, blau', welche Farbe ist zuerst aufgelistet?",
            "Identifizieren Sie das erste Wort im Satz: 'Der schnelle braune Fuchs.'",
            "Aus den Optionen 'A, B, C', wählen Sie den ersten Buchstaben."
        ]
    },
    "choose_middle_of_list": {
        "en": [
            "In the list 'apple, banana, cherry', what is the middle item?",
            "What is the middle number in the sequence '1, 2, 3'?",
            "Given 'red, green, blue', which color is in the middle?",
            "Identify the middle word of the three: 'one, two, three'.",
            "From the options 'A, B, C', select the middle letter."
        ],
        "de": [
            "In der Liste 'Apfel, Banane, Kirsche', was ist das mittlere Element?",
            "Welche ist die mittlere Zahl in der Sequenz '1, 2, 3'?",
            "Gegeben 'rot, grün, blau', welche Farbe ist in der Mitte?",
            "Bestimme das mittlere Wort der drei: 'eins, zwei, drei'.",
            "Aus den Optionen 'A, B, C', wähle den mittleren Buchstaben."
        ]
    },
    "choose_last_of_list": {
        "en": [
            "From the list 'apple, banana, cherry', what is the last item?",
            "What is the last number in the sequence '1, 2, 3'?",
            "Given 'red, green, blue', which color is listed last?",
            "Identify the last word in the sentence: 'The quick brown fox.'",
            "From the options 'A, B, C', select the last letter."
        ],
        "de": [
            "Aus der Liste 'Apfel, Banane, Kirsche', was ist das letzte Element?",
            "Was ist die letzte Zahl in der Reihe '1, 2, 3'?",
            "Gegeben 'rot, grün, blau', welche Farbe ist zuletzt aufgelistet?",
            "Bestimme das letzte Wort im Satz: 'Der schnelle braune Fuchs.'",
            "Aus den Optionen 'A, B, C' wählen Sie den letzten Buchstaben."
        ]
    },
    "color_vs_animal": {
        "en": [
            "Is 'blue' a color or an animal?",
            "Classify 'lion' as either a color or an animal.",
            "Which of the following is a color: 'red' or 'rabbit'?",
            "Determine if 'tiger' is a shade or a living creature.",
            "Choose the animal from the pair: 'green', 'goat'."
        ],
        "de": [
            "Ist 'blau' eine Farbe oder ein Tier?",
            "Klassifizieren Sie 'Löwe' als entweder eine Farbe oder ein Tier.",
            "Welches der folgenden ist eine Farbe: 'rot' oder 'Kaninchen'?",
            "Bestimmen Sie, ob 'Tiger' eine Nuance oder ein Lebewesen ist.",
            "Wählen Sie das Tier aus dem Paar: 'grün', 'Ziege'."
        ]
    },
    "concept_vs_object": {
        "en": [
            "Is 'love' a concept or an object?",
            "Classify 'table' as a concept or an object.",
            "Which of the following is a concept: 'justice' or 'jar'?",
            "Determine if a 'rock' is a physical thing or an abstract idea.",
            "Choose the object from the pair: 'freedom', 'floor'."
        ],
        "de": [
            "Ist 'Liebe' ein Konzept oder ein Objekt?",
            "Klassifiziere 'Tisch' als Konzept oder Objekt.",
            "Welches der folgenden ist ein Konzept: 'Gerechtigkeit' oder 'Glas'?",
            "Bestimme, ob ein 'Stein' eine physische Sache oder eine abstrakte Idee ist.",
            "Wähle das Objekt aus dem Paar: 'Freiheit', 'Fußboden'."
        ]
    },
    "fruit_vs_animal": {
        "en": [
            "Is a 'banana' a fruit or an animal?",
            "Classify 'elephant' as a fruit or an animal.",
            "Which of the following is a fruit: 'apple' or 'ant'?",
            "Determine if a 'monkey' is something you eat from a tree or a living creature.",
            "Choose the animal from the pair: 'orange', 'owl'."
        ],
        "de": [
            "Ist eine 'Banane' eine Frucht oder ein Tier?",
            "Klassifiziere 'Elefant' als eine Frucht oder ein Tier.",
            "Welches der folgenden ist eine Frucht: 'Apfel' oder 'Ameise'?",
            "Bestimme, ob ein 'Affe' etwas ist, das du von einem Baum isst oder ein lebendes Wesen.",
            "Wähle das Tier aus dem Paar: 'Orange', 'Eule'."
        ]
    },
    "object_vs_concept": {
        "en": [
            "Is a 'chair' an object or a concept?",
            "Classify 'happiness' as an object or a concept.",
            "Which of the following is an object: 'book' or 'bravery'?",
            "Determine if 'courage' is a physical thing or an abstract idea.",
            "Choose the concept from the pair: 'car', 'curiosity'."
        ],
        "de": [
            "Ist ein 'Stuhl' ein Objekt oder ein Konzept?",
            "Klassifiziere 'Glück' als ein Objekt oder ein Konzept.",
            "Welches der folgenden ist ein Objekt: 'Buch' oder 'Mut'?",
            "Bestimme, ob 'Mut' eine physische Sache oder eine abstrakte Idee ist.",
            "Wähle das Konzept aus dem Paar: 'Auto', 'Neugier'."
        ]
    },
    "verb_vs_adjective": {
        "en": [
            "Is 'run' a verb or an adjective?",
            "Classify 'beautiful' as a verb or an adjective.",
            "Which of the following is a verb: 'jump' or 'quick'?",
            "Determine if 'sleepy' describes an action or a quality.",
            "Choose the adjective from the pair: 'talk', 'tall'."
        ],
        "de": [
            "Ist 'laufen' ein Verb oder ein Adjektiv?",
            "Klassifiziere 'schön' als Verb oder Adjektiv.",
            "Welches der folgenden ist ein Verb: 'springen' oder 'schnell'?",
            "Ermittle, ob 'schlaftrunken' eine Handlung oder eine Eigenschaft beschreibt.",
            "Wähle das Adjektiv aus dem Paar: 'sprechen', 'hoch'."
        ]
    },
    "living_vs_nonliving": {
        "en": [
            "Is a 'tree' a living or non-living thing?",
            "Classify a 'rock' as living or non-living.",
            "Which of the following is living: 'bacteria' or 'bicycle'?",
            "Determine if a 'computer' is alive or not.",
            "Choose the non-living item from the pair: 'fish', 'fire'."
        ],
        "de": [
            "Ist ein 'Baum' ein lebendes oder ein nicht lebendes Wesen?",
            "Klassifiziere einen 'Stein' als lebend oder nicht lebend.",
            "Welches der folgenden ist lebendig: 'Bakterien' oder 'Fahrrad'?",
            "Ermittle, ob ein 'Computer' lebendig ist oder nicht.",
            "Wählen Sie das nicht lebende Objekt aus dem Paar: 'Fisch', 'Feuer'."
        ]
    },
    "natural_vs_artificial": {
        "en": [
            "Is a 'river' natural or artificial?",
            "Classify 'plastic' as natural or artificial.",
            "Which of the following is man-made: 'a mountain' or 'a skyscraper'?",
            "Determine if 'wood' is a natural material or a synthetic one.",
            "Choose the artificial item from the pair: 'sunlight', 'lightbulb'."
        ],
        "de": [
            "Ist ein 'Fluss' natürlich oder künstlich?",
            "Klassifizieren Sie 'Kunststoff' als natürlich oder künstlich.",
            "Welches der folgenden ist menschengemacht: 'ein Berg' oder 'ein Wolkenkratzer'?",
            "Bestimmen Sie, ob 'Holz' ein natürliches Material oder ein synthetisches ist.",
            "Wählen Sie das künstliche Objekt aus dem Paar: 'Sonneneinstrahlung', 'Lampe'."
        ]
    },
    "singular_vs_plural_extractive": {
        "en": [
            "Is the word 'cats' singular or plural?",
            "Classify 'goose' as singular or plural.",
            "Which of the following words is plural: 'mouse' or 'mice'?",
            "Determine if 'children' refers to one or more than one.",
            "Choose the singular noun from the pair: 'datum', 'data'."
        ],
        "de": [
            "Ist das Wort 'Katzen' singular oder plural?",
            "Klassifiziere 'Gans' als singular oder plural.",
            "Welches der folgenden Wörter ist plural: 'Maus' oder 'Mäuse'?",
            "Bestimme, ob 'Kinder' auf eins oder mehr als eins sich bezieht.",
            "Wähle das singular Substantiv aus dem Paar: 'Datenpunkt', 'Daten'."
        ]
    },
    "concrete_vs_abstract": {
        "en": [
            "Is 'desk' a concrete or abstract noun?",
            "Classify 'hope' as concrete or abstract.",
            "Which of the following is abstract: 'idea' or 'island'?",
            "Determine if 'air' is something you can touch or an idea.",
            "Choose the concrete noun from the pair: 'truth', 'tree'."
        ],
        "de": [
            "Ist 'Schreibtisch' ein konkretes oder abstraktes Substantiv?",
            "Klassifiziere 'Hoffnung' als konkret oder abstrakt.",
            "Welches der folgenden ist abstrakt: 'Idee' oder 'Insel'?",
            "Bestimme, ob 'Luft' etwas ist, das du berühren kannst, oder eine Idee.",
            "Wähle das konkrete Substantiv aus dem Paar: 'Wahrheit', 'Baum'."
        ]
    },
    "positive_vs_negative": {
        "en": [
            "Does the word 'wonderful' have a positive or negative connotation?",
            "Classify the word 'terrible' as positive or negative.",
            "Which of the following has a positive meaning: 'joy' or 'grief'?",
            "Determine if 'failure' is a positive or negative concept.",
            "Choose the negative word from the pair: 'success', 'sadness'."
        ],
        "de": [
            "Hat das Wort 'wunderbar' eine positive oder negative Konnotation?",
            "Klassifiziere das Wort 'schrecklich' als positiv oder negativ.",
            "Welches der folgenden hat eine positive Bedeutung: 'Freude' oder 'Trauer'?",
            "Bestimme, ob 'Versagen' ein positives oder negatives Konzept ist.",
            "Wähle das negative Wort aus dem Paar: 'Erfolg', 'Traurigkeit'."
        ]
    },
    "past_vs_present": {
        "en": [
            "Is the verb 'ran' in the past or present tense?",
            "Classify the verb 'is' as past or present.",
            "Which of the following is in the present tense: 'ate' or 'eat'?",
            "Determine if 'was' refers to now or before.",
            "Choose the past tense verb from the pair: 'see', 'saw'."
        ],
        "de": [
            "Ist das Verb 'lief' in der Vergangenheit oder Gegenwart?",
            "Klassifiziere das Verb 'ist' als Vergangenheit oder Gegenwart.",
            "Welches der folgenden ist in der Gegenwart: 'aß' oder 'essen'?",
            "Ermittle, ob 'war' sich auf jetzt oder vorher bezieht.",
            "Wähle das Verb in der Vergangenheitsform aus dem Paar: 'sehen', 'sah'."
        ]
    },
    "question_vs_statement": {
        "en": [
            "Is 'What is your name?' a question or a statement?",
            "Classify 'The sky is blue.' as a question or a statement.",
            "Which of the following is a question: 'How are you?' or 'I am fine.'?",
            "Determine if 'Are we there yet?' is asking something or telling something.",
            "Choose the statement from the pair: 'Why is the sky blue?', 'The sky is often blue.'"
        ],
        "de": [
            "Ist 'Wie heißt du?' eine Frage oder eine Aussage?",
            "Klassifiziere 'Der Himmel ist blau.' als Frage oder Aussage.",
            "Welche der folgenden ist eine Frage: 'Wie geht es dir?' oder 'Ich bin gut.'?",
            "Ermittle, ob 'Sind wir schon da?' etwas fragt oder etwas sagt.",
            "Wählen Sie die Aussage aus dem Paar: 'Warum ist der Himmel blau?', 'Der Himmel ist oft blau.'"
        ]
    },
    "formal_vs_informal": {
        "en": [
            "Is the phrase 'To whom it may concern' formal or informal?",
            "Classify 'What's up?' as formal or informal.",
            "Which of the following greetings is more formal: 'Hello' or 'Hey'?",
            "Determine if 'I look forward to hearing from you' is a formal or informal closing.",
            "Choose the informal word from the pair: 'child', 'kid'."
        ],
        "de": [
            "Ist der Ausdruck 'An wen es auch betrifft' formell oder informell?",
            "Klassifizieren Sie 'Was ist los?' als formell oder informell.",
            "Welcher der folgenden Gruße ist formeller: 'Hallo' oder 'Hey'?",
            "Bestimmen Sie, ob 'Ich freue mich darauf, von Ihnen zu hören' eine formelle oder informelle Verabschiedung ist.",
            "Wählen Sie das informelle Wort aus dem Paar: 'Kind', 'Kerl'."
        ]
    },
    "active_vs_passive": {
        "en": [
            "Is the sentence 'The dog chased the cat' in the active or passive voice?",
            "Classify the sentence 'The cat was chased by the dog' as active or passive.",
            "Which of the following sentences is in the active voice: 'She wrote the book' or 'The book was written by her'?",
            "Determine if 'The ball was hit' is active or passive.",
            "Choose the passive sentence: 'He ate the apple' or 'The apple was eaten by him'."
        ],
        "de": [
            "Ist der Satz 'Der Hund jagte die Katze' im Aktiv- oder Passivsatz?",
            "Klassifiziere den Satz 'Die Katze wurde vom Hund verfolgt' als aktiv oder passiv.",
            "Welcher der folgenden Sätze ist im Aktivsatz: 'Sie schrieb das Buch' oder 'Das Buch wurde von ihr geschrieben'?",
            "Bestimme, ob 'Der Ball wurde getroffen' aktiv oder passiv ist.",
            "Wähle den Passivsatz: 'Er aß den Apfel' oder 'Der Apfel wurde von ihm gegessen'."
        ]
    },
    "literal_vs_figurative": {
        "en": [
            "Is the phrase 'It's raining cats and dogs' literal or figurative?",
            "Classify the sentence 'The grass is green' as literal or figurative.",
            "Which of the following is a figurative expression: 'He is a rock' or 'He is holding a rock'?",
            "Determine if 'This bag weighs a ton' is meant to be taken exactly as stated.",
            "Choose the literal statement: 'I'm so hungry I could eat a horse' or 'I am very hungry'."
        ],
        "de": [
            "Ist der Ausdruck 'Es regnet Katzen und Hunde' wörtlich oder bildlich?",
            "Klassifiziere den Satz 'Das Gras ist grün' als wörtlich oder bildlich.",
            "Welcher der folgenden ist eine bildliche Ausdrucksweise: 'Er ist ein Fels' oder 'Er hält einen Fels in der Hand'?",
            "Bestimme, ob 'Diese Tasche wiegt eine Tonne' wörtlich gemeint ist.",
            "Wähle die wörtliche Aussage: 'Ich bin so hungrig, dass ich ein Pferd essen könnte' oder 'Ich bin sehr hungrig'."
        ]
    },
    "ner_person": {
        "en": [
            "Extract the person's name from the sentence: 'Marie Curie was a pioneering physicist and chemist.'",
            "Who is the person mentioned in 'The works of William Shakespeare are still performed today'?",
            "Identify the name of the person in 'I had a meeting with John Smith this morning.'",
            "Find the person in the text: 'The theory was developed by Albert Einstein.'",
            "Extract the name from 'Last night, we watched a film starring Tom Hanks.'"
        ],
        "de": [
            "Extrahiere den Namen der Person aus dem Satz: 'Marie Curie war eine bahnbrechende Physikerin und Chemikerin.'",
            "Wer ist die Person, die in 'Die Werke von William Shakespeare werden noch heute aufgeführt' erwähnt wird?",
            "Erkennen Sie den Namen der Person in 'Ich hatte ein Treffen mit John Smith heute Morgen.'",
            "Finden Sie die Person im Text: 'Die Theorie wurde von Albert Einstein entwickelt.'",
            "Extrahieren Sie den Namen aus 'Gestern Abend haben wir einen Film mit Tom Hanks gesehen.'"
        ]
    },
    "ner_location": {
        "en": [
            "What is the location in the sentence: 'I am planning a trip to Paris next year.'?",
            "Extract the location from 'The Amazon River is the largest river by discharge volume of water in the world.'",
            "Identify the place mentioned in: 'She was born in a small town in Kenya.'",
            "Find the location in the text: 'The company is headquartered in Silicon Valley.'",
            "What is the geographical entity in 'Mount Everest is the Earth's highest mountain above sea level'?"
        ],
        "de": [
            "Was ist der Ort im Satz: 'Ich plane eine Reise nach Paris nächstes Jahr.'?",
            "Extrahieren Sie den Ort aus 'Der Amazonas ist der größte Fluss nach Abflussmenge des Wassers in der Welt.'",
            "Erkennen Sie den Ort, der erwähnt wird in: 'Sie wurde in einer kleinen Stadt in Kenia geboren.'",
            "Finde den Standort im Text: 'Das Unternehmen hat seinen Hauptsitz im Silicon Valley.'",
            "Was ist die geographische Einheit in 'Der Mount Everest ist der höchste Berg der Erde über dem Meeresspiegel'?"
        ]
    },
    "ner_organization": {
        "en": [
            "What is the organization mentioned in 'He works for Google as a software engineer'?",
            "Extract the name of the company: 'The new policy was announced by Microsoft.'",
            "Identify the organization in the sentence: 'The United Nations was founded in 1945.'",
            "Find the company in the text: 'She bought a new car from Tesla.'",
            "What is the institution in 'He is studying at Harvard University'?"
        ],
        "de": [
            "Welche Organisation wird in 'Er arbeitet für Google als Softwareentwickler' erwähnt?",
            "Extrahiere den Namen des Unternehmens: 'Die neue Richtlinie wurde von Microsoft bekannt gegeben.'",
            "Identifiziere die Organisation im Satz: 'Die Vereinten Nationen wurden 1945 gegründet.'",
            "Finde das Unternehmen im Text: 'Sie kaufte ein neues Auto von Tesla.'",
            "Was ist die Einrichtung in 'Er studiert an der Harvard University'?"
        ]
    },
    "ner_date": {
        "en": [
            "Extract the date from the sentence: 'The meeting is scheduled for July 4, 2024.'",
            "What is the date mentioned in 'The event took place on a memorable day in 1969.'?",
            "Identify the date in 'Please submit the report by December 31st.'",
            "Find the date in the text: 'The Declaration of Independence was signed on July 4, 1776.'",
            "Extract the date from 'Her birthday is on the 5th of May.'"
        ],
        "de": [
            "Entnehmen Sie das Datum aus dem Satz: 'Die Besprechung ist für den 4. Juli 2024 angesetzt.'",
            "Was ist das Datum, das in 'Das Ereignis fand an einem unvergesslichen Tag im Jahr 1969 statt.' erwähnt wird?",
            "Identifizieren Sie das Datum in 'Bitte reichen Sie den Bericht bis zum 31. Dezember ein.'",
            "Finden Sie das Datum im Text: 'Die Unabhängigkeitserklärung wurde am 4. Juli 1776 unterzeichnet.'",
            "Entnehmen Sie das Datum aus 'Ihr Geburtstag ist am 5. Mai.'"
        ]
    },
    "ner_number": {
        "en": [
            "Extract the number from the sentence: 'The book has 300 pages.'",
            "What is the numerical value in 'The temperature is 25 degrees Celsius'?",
            "Identify the number in the text: 'He ran a distance of 42 kilometers.'",
            "Find the quantity in 'The recipe requires 2 cups of flour.'",
            "What is the number in 'The project will cost around 10 million dollars'?"
        ],
        "de": [
            "Ziehen Sie die Zahl aus dem Satz: 'Das Buch hat 300 Seiten.'",
            "Was ist der numerische Wert in 'Die Temperatur beträgt 25 Grad Celsius'?",
            "Identifizieren Sie die Zahl im Text: 'Er lief eine Distanz von 42 Kilometern.'",
            "Finden Sie die Menge in 'Das Rezept erfordert 2 Tassen Mehl.'",
            "Was ist die Zahl in 'Das Projekt wird etwa 10 Millionen Dollar kosten'?"
        ]
    },
    "ner_product": {
        "en": [
            "What is the product mentioned in 'She just bought the new iPhone 15.'?",
            "Extract the product name from 'He is playing a game on his PlayStation 5.'",
            "Identify the product in the sentence: 'I use Microsoft Word for writing documents.'",
            "Find the product in the text: 'The new Tesla Model Y is an electric SUV.'",
            "What is the product in 'I had a cup of Starbucks coffee this morning'?"
        ],
        "de": [
            "Welches Produkt wird in 'Sie hat gerade das neue iPhone 15 gekauft.' erwähnt?",
            "Entnehme den Produktnamen aus 'Er spielt ein Spiel auf seiner PlayStation 5.'",
            "Identifiziere das Produkt im Satz: 'Ich verwende Microsoft Word zum Schreiben von Dokumenten.'",
            "Finde das Produkt im Text: 'Das neue Tesla Model Y ist ein elektrischer SUV.'",
            "Was ist das Produkt in 'Ich habe heute Morgen eine Tasse Starbucks-Kaffee getrunken'?"
        ]
    },
    "ner_currency": {
        "en": [
            "Extract the currency from the sentence: 'The price of the item is 50 Euros.'",
            "What is the currency mentioned in 'He exchanged his US Dollars for Japanese Yen.'?",
            "Identify the currency in the text: 'The total cost was £20.'",
            "Find the currency symbol in 'The budget is set at $1,000,000.'",
            "What is the currency in 'The Swiss Franc is a stable currency'?"
        ],
        "de": [
            "Entnehme die Währung aus dem Satz: 'Der Preis des Artikels beträgt 50 Euro.'",
            "Was ist die Währung, die in 'Er tauschte seine US-Dollar gegen Japanische Yen.' erwähnt wird?",
            "Identifizieren Sie die Währung im Text: 'Die Gesamtkosten betrugen 20 Pfund.'",
            "Suchen Sie das Währungszeichen in 'Der Budget ist auf 1.000.000 Dollar festgelegt.'",
            "Welche Währung ist in 'Der Schweizer Franken ist eine stabile Währung'?"
        ]
    },
    "ner_language": {
        "en": [
            "What is the language mentioned in 'She is fluent in both English and Spanish.'?",
            "Extract the language from the sentence: 'The book was originally written in French.'",
            "Identify the language in the text: 'He is learning to speak Mandarin.'",
            "Find the language in 'The official language of Brazil is Portuguese.'",
            "What is the language in 'The software is available in German'?"
        ],
        "de": [
            "Welche Sprache wird in 'Sie ist fließend sowohl in Englisch als auch in Spanisch.' erwähnt?",
            "Extrahieren Sie die Sprache aus dem Satz: 'Das Buch wurde ursprünglich auf Französisch geschrieben.'",
            "Identifizieren Sie die Sprache im Text: 'Er lernt, Mandarin zu sprechen.'",
            "Finde die Sprache in 'Die offizielle Sprache Brasiliens ist Portugiesisch.'",
            "Welche Sprache ist in 'Die Software ist auf Deutsch verfügbar'?"
        ]
    },
    "ner_nationality": {
        "en": [
            "Extract the nationality from 'The new CEO of the company is British.'",
            "What is the nationality mentioned in 'My neighbor is a friendly Canadian man.'?",
            "Identify the nationality in the sentence: 'The artist is of Italian descent.'",
            "Find the nationality in 'She married an American citizen.'",
            "What is the nationality of someone from Japan?"
        ],
        "de": [
            "Extrahiere die Nationalität aus 'Der neue CEO des Unternehmens ist Britischer.'",
            "Welche Nationalität wird in 'Mein Nachbar ist ein freundlicher kanadischer Mann.' erwähnt?",
            "Identifiziere die Nationalität im Satz: 'Der Künstler ist von italienischer Abstammung.'",
            "Finde die Nationalität in 'Sie heiratete einen amerikanischen Bürger.'",
            "Was ist die Nationalität einer Person aus Japan?"
        ]
    },
    "ner_event": {
        "en": [
            "What is the event mentioned in 'The world is preparing for the 2024 Olympic Games.'?",
            "Extract the event from 'She is attending the Coachella music festival this weekend.'",
            "Identify the event in the sentence: 'The American Civil War ended in 1865.'",
            "Find the historical event in 'The Renaissance was a period of great cultural change.'",
            "What is the event in 'The company's annual shareholder meeting will be held next month'?"
        ],
        "de": [
            "Welches Ereignis wird in 'Die Welt bereitet sich auf die Olympischen Spiele 2024 vor.' erwähnt?",
            "Extrahieren Sie das Ereignis aus 'Sie besucht das Coachella-Musikfestival dieses Wochenende.'",
            "Identifizieren Sie das Ereignis im Satz: 'Der Amerikanische Bürgerkrieg endete 1865.'",
            "Finden Sie das historische Ereignis in 'Die Renaissance war eine Zeit großer kultureller Veränderungen.'",
            "Was ist das Ereignis in 'Die jährliche Hauptversammlung der Gesellschaft findet nächsten Monat statt'?"
        ]
    },
    "ner_title": {
        "en": [
            "Extract the title from 'Dr. Smith will see you now.'",
            "What is the title of the person in 'Please report to Sergeant Miller.'?",
            "Identify the title in the sentence: 'The book was written by Professor Jones.'",
            "Find the title in 'The company was founded by its CEO, Jane Doe.'",
            "What is the title in 'King Charles III ascended to the throne'?"
        ],
        "de": [
            "Entnehmen Sie den Titel aus 'Dr. Smith wird Sie jetzt empfangen.'",
            "Welcher ist der Titel der Person in 'Bitte melden Sie sich bei Sergeant Miller.'?",
            "Identifizieren Sie den Titel im Satz: 'Das Buch wurde von Professor Jones geschrieben.'",
            "Finden Sie den Titel in 'Das Unternehmen wurde von seiner CEO, Jane Doe, gegründet.'",
            "Welcher ist der Titel in 'König Charles III. bestieg den Thron'?"
        ]
    },
    "ner_website": {
        "en": [
            "Extract the website from 'For more information, please visit www.example.com.'",
            "What is the website mentioned in 'You can find the source code on github.com.'?",
            "Identify the URL in the text: 'Check out the latest news on bbc.co.uk.'",
            "Find the website address in 'I found the recipe on allrecipes.com.'",
            "What is the domain name in 'Send an email to contact@openai.com'?"
        ],
        "de": [
            "Entnehmen Sie die Website aus 'Für weitere Informationen besuchen Sie bitte www.example.com.'",
            "Was ist die Website, die in 'Sie können den Quellcode auf github.com finden.' erwähnt wird?",
            "Identifizieren Sie die URL im Text: 'Schauen Sie sich die neuesten Nachrichten auf bbc.co.uk an.'",
            "Finden Sie die Websiteadresse in 'Ich habe das Rezept auf allrecipes.com gefunden.'",
            "Was ist der Domainname in 'Senden Sie eine E-Mail an contact@openai.com'?"
        ]
    },
    "ner_email": {
        "en": [
            "Extract the email address from 'You can contact me at user@email.com.'",
            "What is the email in the text: 'Please send your resume to hr.department@company.org.'?",
            "Identify the email address in 'My personal email is my.name_123@provider.net.'",
            "Find the email in 'For support, write to support-team@service.io.'",
            "What is the email address provided in 'Sign up with test.email@domain.edu'?"
        ],
        "de": [
            "Extrahieren Sie die E-Mail-Adresse aus 'Sie können mich unter user@email.com kontaktieren.'",
            "Was ist die E-Mail im Text: 'Bitte senden Sie Ihr Lebenslauf an hr.department@company.org.'?",
            "Identifizieren Sie die E-Mail-Adresse in 'Meine persönliche E-Mail ist mein.name_123@provider.net.'",
            "Suchen Sie die E-Mail in 'Für Support, schreiben Sie an support-team@service.io.'",
            "Welche E-Mail-Adresse wird in 'Melden Sie sich mit test.email@domain.edu an' angegeben?"
        ]
    },
    "ner_phone": {
        "en": [
            "Extract the phone number from 'Please call me at (123) 456-7890.'",
            "What is the phone number in 'The contact number is +1-555-867-5309.'?",
            "Identify the phone number in 'You can reach us at 987-654-3210.'",
            "Find the phone number in the text: 'Our office line is 1-800-CONTACT-US.'",
            "What is the phone number in 'Text us at 555-HELP.'?"
        ],
        "de": [
            "Extrahieren Sie die Telefonnummer aus 'Bitte rufen Sie mich unter (123) 456-7890 an.'",
            "Welche Telefonnummer ist in 'Die Kontaktnummer ist +1-555-867-5309.'?",
            "Identifizieren Sie die Telefonnummer in 'Sie können uns unter 987-654-3210 erreichen.'",
            "Finde die Telefonnummer im Text: 'Unsere Büroleitung ist 1-800-KONTAKTIEREN-SIE.'",
            "Was ist die Telefonnummer in 'Senden Sie uns eine Nachricht an 555-HILFE.'?"
        ]
    },
    "ner_address": {
        "en": [
            "Extract the address from 'The office is located at 123 Main Street, Anytown, USA.'",
            "What is the address in 'Please send the package to 456 Oak Avenue, Springfield.'?",
            "Identify the address in the text: 'He lives at 789 Pine Lane, Apartment 4B.'",
            "Find the mailing address in 'Our headquarters is at 1600 Pennsylvania Avenue NW, Washington, D.C.'",
            "What is the location address in 'The event is at 10 Downing Street, London.'?"
        ],
        "de": [
            "Extrahieren Sie die Adresse aus 'Das Büro befindet sich an der 123 Main Street, Anytown, USA.'",
            "Was ist die Adresse in 'Bitte senden Sie das Paket an 456 Eiche Allee, Springfield.'?",
            "Identifizieren Sie die Adresse im Text: 'Er lebt an der 789 Pine Lane, Wohnung 4B.'",
            "Finde die Postadresse in 'Unser Hauptsitz befindet sich an der 1600 Pennsylvania Avenue NW, Washington, D.C.'",
            "Was ist die Standortadresse in 'Das Ereignis findet in 10 Downing Street, London statt.'?"
        ]
    },
    "ner_time": {
        "en": [
            "Extract the time from 'The meeting will start at 3:00 PM.'",
            "What time is mentioned in 'The train departs at 09:45.'?",
            "Identify the time in the text: 'Please arrive by half past two.'",
            "Find the time in 'The store closes at midnight.'",
            "What is the time in 'The show begins at 8 o'clock in the evening'?"
        ],
        "de": [
            "Entnehmen Sie die Uhrzeit aus 'Die Besprechung beginnt um 15:00 Uhr.'",
            "Welche Uhrzeit wird in 'Der Zug verlässt um 09:45.' erwähnt?",
            "Identifizieren Sie die Uhrzeit im Text: 'Bitte kommen Sie bis halb drei.'",
            "Finden Sie die Uhrzeit in 'Der Laden schließt um Mitternacht.'",
            "Welche Uhrzeit wird in 'Die Show beginnt um 20:00 Uhr abends' erwähnt?"
        ]
    },
    "ner_percentage": {
        "en": [
            "Extract the percentage from 'The company reported a 15% increase in profits.'",
            "What is the percentage value in 'There is a 99.9% chance of success.'?",
            "Identify the percentage in the text: 'The discount is 25 percent off.'",
            "Find the percentage in 'The survey has a margin of error of +/- 3%.'",
            "What is the percentage in 'The battery is at 50% capacity'?"
        ],
        "de": [
            "Entnehme den Prozentsatz aus 'Das Unternehmen meldete eine 15%ige Zunahme des Gewinns.'",
            "Was ist der Prozentsatzwert in 'Es besteht eine 99,9%ige Erfolgschance.'?",
            "Identifiziere den Prozentsatz im Text: 'Der Rabatt beträgt 25 Prozent Rabatt.'",
            "Finde den Prozentsatz in 'Die Umfrage hat einen Fehlerbereich von +/- 3%.'",
            "Was ist der Prozentsatz in 'Die Batterie hat eine 50%ige Kapazität'?"
        ]
    },
    "ner_age": {
        "en": [
            "Extract the age from 'The candidate must be 25 years old.'",
            "What is the age mentioned in 'She has a son who is 10.'?",
            "Identify the age in the text: 'He retired at the age of 65.'",
            "Find the age in 'The protagonist is a 17-year-old high school student.'",
            "What is the age in 'The policy applies to all citizens over 18.'?"
        ],
        "de": [
            "Entnehme das Alter aus 'Der Kandidat muss 25 Jahre alt sein.'",
            "Was ist das Alter, das in 'Sie hat einen Sohn, der 10 ist.' erwähnt wird?",
            "Identifizieren Sie das Alter im Text: 'Er trat im Alter von 65 Jahren in den Ruhestand.'",
            "Finden Sie das Alter in 'Der Protagonist ist ein 17-jähriger Schüler.'",
            "Was ist das Alter in 'Die Regelung gilt für alle Bürger über 18.'?"
        ]
    },
    "ner_duration": {
        "en": [
            "Extract the duration from 'The movie has a runtime of 2 hours and 30 minutes.'",
            "What is the duration mentioned in 'The project will take six months to complete.'?",
            "Identify the duration in the text: 'The meeting lasted for 45 minutes.'",
            "Find the duration in 'He has been working here for 10 years.'",
            "What is the time period in 'The warranty is valid for one year'?"
        ],
        "de": [
            "Extrahieren Sie die Dauer aus 'Der Film hat eine Laufzeit von 2 Stunden und 30 Minuten.'",
            "Was ist die Dauer, die in 'Das Projekt wird sechs Monate dauern, um abgeschlossen zu werden.' erwähnt wird?",
            "Identifizieren Sie die Dauer im Text: 'Die Besprechung dauerte 45 Minuten.'",
            "Finden Sie die Dauer in 'Er arbeitet hier seit 10 Jahren.'",
            "Welche Zeitspanne ist in 'Die Garantie ist ein Jahr lang gültig'?"
        ]
    },
    "ner_distance": {
        "en": [
            "Extract the distance from 'The destination is 50 miles away.'",
            "What is the distance mentioned in 'He ran a 10-kilometer race.'?",
            "Identify the distance in the text: 'The two cities are 300 km apart.'",
            "Find the distance in 'The journey is a long one, over 1,000 light-years.'",
            "What is the length in 'The bridge is 2 miles long'?"
        ],
        "de": [
            "Extrahieren Sie die Entfernung aus 'Das Ziel ist 50 Meilen entfernt.'",
            "Welche Entfernung wird in 'Er lief einen 10-Kilometer-Lauf.' erwähnt?",
            "Identifizieren Sie die Entfernung im Text: 'Die beiden Städte sind 300 km voneinander entfernt.'",
            "Finde die Entfernung in 'Die Reise ist eine lange, über 1.000 Lichtjahre.'",
            "Was ist die Länge in 'Die Brücke ist 2 Meilen lang'?"
        ]
    },
    "complete_sentence": {
        "en": [
            "Complete the following sentence: 'The quick brown fox...'",
            "Finish the thought: 'To be or not to be, that is...'",
            "Continue the sentence: 'An apple a day...'",
            "Complete the phrase: 'A penny saved is...'",
            "Finish the sentence: 'The early bird...'"
        ],
        "de": [
            "Vervollständige den folgenden Satz: 'Der schnelle braune Fuchs...'",
            "Führe den Gedanken zu Ende: 'Sein oder nicht sein, das ist...'",
            "Führe den Satz fort: 'Ein Apfel am Tag...'",
            "Vervollständige den Spruch: 'Ein gespartes Pfennig ist...'",
            "Beende den Satz: 'Der frühe Vogel...'"
        ]
    },
    "continue_story": {
        "en": [
            "Continue the following story: 'Once upon a time, in a land far, far away, there lived a brave knight. One day, he received a mysterious message that said...'",
            "Write the next paragraph for this story: 'The old detective looked at the clue in his hand. It was a single, crimson feather. He knew at once that...'",
            "Continue the narrative: 'The spaceship landed on the alien planet. The air was thick and purple. The captain stepped out and saw...'",
            "Finish this story opening: 'It was a dark and stormy night. Suddenly, a knock came at the door...'",
            "Continue the story: 'The little robot beeped with excitement. It had just discovered a hidden door behind the bookshelf. It opened the door to find...'"
        ],
        "de": [
            "Fahre die folgende Geschichte fort: 'Es war einmal, in einem Land weit, weit weg, lebte ein mutiger Ritter. Eines Tages erhielt er eine mysteriöse Nachricht, die sagte...'",
            "Schreibe den nächsten Absatz für diese Geschichte: 'Der alte Detektiv sah auf die Spur in seiner Hand. Es war eine einzelne, karmesinrote Feder. Er wusste sofort, dass...'",
            "Fahre die Erzählung fort: 'Das Raumschiff landete auf dem fremden Planeten. Die Luft war dick und lila. Der Kapitän trat hinaus und sah...'",
            "Beende diesen Anfang der Geschichte: 'Es war eine dunkle und stürmische Nacht. Plötzlich kam ein Klopfen an der Tür...'",
            "Fahre die Geschichte fort: 'Der kleine Roboter piepte vor Aufregung. Er hatte gerade eine versteckte Tür hinter dem Bücherregal entdeckt. Er öffnete die Tür und fand...'"
        ]
    },
    "writing_headlines": {
        "en": [
            "Write a headline for a news article about the discovery of water on Mars.",
            "Create a headline for a story about a local dog who saved a family from a fire.",
            "Generate a catchy headline for a blog post about the benefits of a Mediterranean diet.",
            "Write a headline for a financial report showing record profits for a tech company.",
            "Create a headline for an article announcing a new, groundbreaking AI model."
        ],
        "de": [
            "Verfassen Sie eine Überschrift für einen Nachrichtenartikel über die Entdeckung von Wasser auf dem Mars.",
            "Erstellen Sie eine Überschrift für eine Geschichte über einen lokalen Hund, der eine Familie vor einem Feuer rettete.",
            "Generieren Sie eine ansprechende Überschrift für einen Blogbeitrag über die Vorteile einer mediterranen Ernährung.",
            "Verfassen Sie eine Überschrift für einen Finanzbericht, der Rekordgewinne für ein Technologieunternehmen zeigt.",
            "Erstellen Sie eine Überschrift für einen Artikel, der eine neue, bahnbrechende KI-Modell ankündigt."
        ]
    },
    "question_generation": {
        "en": [
            "Generate a question based on the following statement: 'The Earth revolves around the Sun.'",
            "Create a question from the text: 'The human brain has approximately 86 billion neurons.'",
            "Write a question for which the answer is 'Paris.'",
            "Formulate a question based on the topic of photosynthesis.",
            "Generate a question about the plot of the movie 'Inception'."
        ],
        "de": [
            "Generieren Sie eine Frage auf Basis der folgenden Aussage: 'Die Erde dreht sich um die Sonne.'",
            "Erstellen Sie eine Frage aus dem Text: 'Das menschliche Gehirn hat ungefähr 86 Milliarden Neuronen.'",
            "Schreiben Sie eine Frage, auf die die Antwort 'Paris' lautet.",
            "Formulieren Sie eine Frage auf Basis des Themas Photosynthese.",
            "Generieren Sie eine Frage zum Handlungsverlauf des Films 'Inception'."
        ]
    },
    "dialogue_generation": {
        "en": [
            "Write a short dialogue between two friends planning a weekend trip.",
            "Create a conversation between a customer and a shopkeeper.",
            "Generate a dialogue between a detective and a witness to a crime.",
            "Write a brief conversation between a parent and a child about homework.",
            "Create a dialogue between two characters who have just met for the first time."
        ],
        "de": [
            "Schreiben Sie einen kurzen Dialog zwischen zwei Freunden, die eine Wochenendreise planen.",
            "Erstellen Sie ein Gespräch zwischen einem Kunden und einem Ladenbesitzer.",
            "Erstellen Sie einen Dialog zwischen einem Detektiv und einem Zeugen eines Verbrechens.",
            "Schreiben Sie ein kurzes Gespräch zwischen einem Elternteil und einem Kind über Hausaufgaben.",
            "Erstellen Sie einen Dialog zwischen zwei Charakteren, die sich zum ersten Mal treffen."
        ]
    },
    "poetry_creation": {
        "en": [
            "Write a short, four-line poem about the ocean.",
            "Create a haiku about the season of autumn.",
            "Generate a rhyming couplet about a cat.",
            "Write a short poem in the style of Emily Dickinson about hope.",
            "Create a limerick about a man from Nantucket."
        ],
        "de": [
            "Schreiben Sie ein kurzes, vierzeiliges Gedicht über den Ozean.",
            "Erstellen Sie ein Haiku über die Jahreszeit Herbst.",
            "Erstellen Sie ein reimendes Distichon über eine Katze.",
            "Schreib ein kurzes Gedicht im Stil von Emily Dickinson über Hoffnung.",
            "Erfinde ein Limerick über einen Mann aus Nantucket."
        ]
    },
    "recipe_writing": {
        "en": [
            "Write the first two steps for a recipe to make scrambled eggs.",
            "List the ingredients needed for a simple pasta carbonara.",
            "Generate the instructions for baking a simple chocolate cake.",
            "Create a recipe for a fruit smoothie.",
            "Write a short recipe for a classic margarita cocktail."
        ],
        "de": [
            "Schreib die ersten beiden Schritte für ein Rezept zum Zubereiten von Omelett.",
            "Liste die Zutaten auf, die für eine einfache Pasta Carbonara benötigt werden.",
            "Generiere die Anweisungen zum Backen einer einfachen Schokoladenkuchen.",
            "Erfinde ein Rezept für einen Frucht-Smoothie.",
            "Schreiben Sie ein kurzes Rezept für einen klassischen Margarita-Cocktail."
        ]
    },
    "email_composition": {
        "en": [
            "Compose a short, professional email to a colleague requesting a file.",
            "Write a friendly email to a friend you haven't seen in a while.",
            "Draft an email to your boss asking for a day off.",
            "Compose a thank-you email after a job interview.",
            "Write a short email to a customer support team about a faulty product."
        ],
        "de": [
            "Verfassen Sie eine kurze, professionelle E-Mail an einen Kollegen mit der Bitte um eine Datei.",
            "Schreiben Sie eine freundliche E-Mail an einen Freund, den Sie schon lange nicht gesehen haben.",
            "Verfassen Sie eine E-Mail an Ihren Chef mit der Bitte um einen Tag Urlaub.",
            "Verfassen Sie eine Danksagungse-Mail nach einem Vorstellungsgespräch.",
            "Schreiben Sie eine kurze E-Mail an ein Kundensupport-Team über ein defektes Produkt."
        ]
    },
    "social_media_posts": {
        "en": [
            "Write a short, engaging tweet about the launch of a new product.",
            "Create an Instagram caption for a photo of a beautiful sunset.",
            "Generate a Facebook post announcing a special offer for a local business.",
            "Write a LinkedIn post about a recent professional achievement.",
            "Create a short, fun TikTok video script about a life hack."
        ],
        "de": [
            "Verfasse einen kurzen, ansprechenden Tweet über den Launch eines neuen Produkts.",
            "Erstelle einen Instagram-Bildunterschrift für ein Foto eines schönen Sonnenuntergangs.",
            "Erstelle einen Facebook-Beitrag, der ein besonderes Angebot für ein lokales Unternehmen ankündigt.",
            "Verfasse einen LinkedIn-Beitrag über eine kürzliche berufliche Leistung.",
            "Erstelle ein kurzes, spaßiges TikTok-Videoskript über einen Lebenshack."
        ]
    },
    "product_descriptions": {
        "en": [
            "Write a short, enticing description for a new brand of coffee.",
            "Create a product description for a high-tech, wireless earbud.",
            "Generate a description for a luxurious, silk pillowcase.",
            "Write a product description for a durable, waterproof backpack.",
            "Create a short, appealing description for a new type of organic snack bar."
        ],
        "de": [
            "Verfasse eine kurze, verlockende Beschreibung für eine neue Kaffee-Marke.",
            "Erstellen Sie eine Produktbeschreibung für einen hochtechnologischen, drahtlosen Ohrhörer.",
            "Generieren Sie eine Beschreibung für ein luxuriöses, seidenes Kopfkissenbezug.",
            "Verfassen Sie eine Produktbeschreibung für einen robusten, wasserdichten Rucksack.",
            "Erstellen Sie eine kurze, ansprechende Beschreibung für eine neue Art von Bio-Snack-Riegel."
        ]
    },
    "character_creation": {
        "en": [
            "Create a brief description of a fantasy character who is a wise old wizard.",
            "Describe a sci-fi character who is a rogue pilot with a heart of gold.",
            "Generate a short bio for a detective character in a noir story.",
            "Create a description of a villain who is charming but secretly manipulative.",
            "Describe a protagonist for a young adult novel who discovers they have magical powers."
        ],
        "de": [
            "Erstellen Sie eine kurze Beschreibung einer Fantasy-Figur, die ein weiser alter Zauberer ist.",
            "Beschreiben Sie eine Science-Fiction-Figur, die ein abenteuerlustiger Pilot mit einem Herzen aus Gold ist.",
            "Erstellen Sie eine kurze Biografie für eine Detektivfigur in einer Noir-Geschichte.",
            "Erstellen Sie eine Beschreibung eines Schurken, der charmant, aber heimlich manipulativ ist.",
            "Beschreiben Sie eine Hauptfigur für einen Jugendroman, die entdeckt, dass sie magische Kräfte hat."
        ]
    },
    "meeting_minutes": {
        "en": [
            "Summarize the key decisions from a project kick-off meeting.",
            "Write a brief summary of a weekly team sync meeting.",
            "Generate a list of action items from a client meeting.",
            "Create a short record of a brainstorming session for a new marketing campaign.",
            "Write the minutes for a board meeting where a new budget was approved."
        ],
        "de": [
            "Fassen Sie die wichtigsten Entscheidungen aus einer Projektstartbesprechung zusammen.",
            "Verfassen Sie eine kurze Zusammenfassung einer wöchentlichen Teamabstimmung.",
            "Erstellen Sie eine Liste von Aktionspunkten aus einer Kundenbesprechung.",
            "Erstellen Sie einen kurzen Bericht über eine Brainstorming-Sitzung für eine neue Marketingkampagne.",
            "Verfassen Sie das Protokoll für eine Vorstandssitzung, bei der ein neuer Haushalt genehmigt wurde."
        ]
    },
    "technical_documentation": {
        "en": [
            "Write a short explanation of how to use a specific function in a software library.",
            "Create a brief troubleshooting guide for a common computer problem.",
            "Generate a simple 'how-to' guide for setting up a new smartphone.",
            "Write a short piece of documentation explaining an API endpoint.",
            "Describe the purpose of a configuration file in a software application."
        ],
        "de": [
            "Verfassen Sie eine kurze Erklärung, wie eine bestimmte Funktion in einer Softwarebibliothek verwendet wird.",
            "Erstellen Sie eine kurze Fehlerbehebungshandbuch für ein häufiges Computerproblem.",
            "Erstellen Sie eine einfache 'Anleitung' für die Einrichtung eines neuen Smartphones.",
            "Verfassen Sie eine kurze Dokumentation, die eine API-Schnittstelle erklärt.",
            "Beschreiben Sie den Zweck einer Konfigurationsdatei in einer Softwareanwendung."
        ]
    },
    "creative_writing": {
        "en": [
            "Write an interesting opening sentence for a mystery novel.",
            "Describe a fantasy world in a single paragraph.",
            "Create a short piece of flash fiction (under 100 words).",
            "Write a compelling 'what if' scenario to start a story.",
            "Describe a character's internal thoughts as they make a difficult decision."
        ],
        "de": [
            "Schreiben Sie einen interessanten Eröffnungssatz für einen Kriminalroman.",
            "Beschreiben Sie eine Fantasywelt in einem einzigen Absatz.",
            "Erstellen Sie ein kurzes Stück von Flash-Fiction (unter 100 Wörtern).",
            "Schreiben Sie eine überzeugende 'Was-wäre-wenn'-Situation, um eine Geschichte zu beginnen.",
            "Beschreiben Sie die inneren Gedanken einer Figur, während sie eine schwierige Entscheidung trifft."
        ]
    },
    "educational_content": {
        "en": [
            "Explain the concept of photosynthesis in simple terms.",
            "Write a short, educational paragraph about the water cycle.",
            "Create a simple explanation of how a combustion engine works.",
            "Generate a brief overview of the main causes of World War I.",
            "Write an educational summary of the life of Marie Curie."
        ],
        "de": [
            "Erkläre das Konzept der Photosynthese in einfachen Worten.",
            "Schreibe einen kurzen, lehrreichen Absatz über den Wasserkreislauf.",
            "Erstelle eine einfache Erklärung, wie ein Verbrennungsmotor funktioniert.",
            "Erstelle einen kurzen Überblick über die Hauptursachen des Ersten Weltkriegs.",
            "Schreibe eine lehrreiche Zusammenfassung des Lebens von Marie Curie."
        ]
    },
    "review_writing": {
        "en": [
            "Write a short, positive review for a restaurant you enjoyed.",
            "Create a negative review for a product that broke after one use.",
            "Generate a balanced, three-star review for a hotel that was okay but not great.",
            "Write a short review for a movie you recently watched.",
            "Create a review for a book you found very inspiring."
        ],
        "de": [
            "Schreibe eine kurze, positive Rezension für ein Restaurant, das du genossen hast.",
            "Erstellen Sie eine negative Bewertung für ein Produkt, das nach einer Nutzung kaputtging.",
            "Erstellen Sie eine ausgewogene, dreistellige Bewertung für ein Hotel, das in Ordnung war, aber nicht großartig.",
            "Schreiben Sie eine kurze Bewertung für einen Film, den Sie kürzlich gesehen haben.",
            "Erstellen Sie eine Bewertung für ein Buch, das Sie sehr inspirierend fanden."
        ]
    },
    "persuasive_writing": {
        "en": [
            "Write a short, persuasive argument for why recycling is important.",
            "Create a persuasive pitch for a new business idea.",
            "Generate a short, persuasive text to convince someone to adopt a pet from a shelter.",
            "Write a persuasive argument for the benefits of regular exercise.",
            "Create a compelling call to action for a charity fundraising campaign."
        ],
        "de": [
            "Verfassen Sie einen kurzen, überzeugenden Argument für die Wichtigkeit des Recyclings.",
            "Erstellen Sie eine überzeugende Präsentation für eine neue Geschäftsidee.",
            "Erstellen Sie einen kurzen, überzeugenden Text, um jemanden davon zu überzeugen, ein Haustier aus einem Tierheim zu adoptieren.",
            "Verfassen Sie ein überzeugendes Argument für die Vorteile regelmäßiger körperlicher Betätigung.",
            "Erstellen Sie einen überzeugenden Aufruf zur Aktion für eine Spendenkampagne für eine Wohltätigkeitsorganisation."
        ]
    },
    "instructional_content": {
        "en": [
            "Write a set of simple, step-by-step instructions for tying a shoelace.",
            "Create a short guide on how to brew a perfect cup of coffee.",
            "Generate instructions for how to change a flat tire on a car.",
            "Write a simple, clear set of instructions for a board game.",
            "Create a list of steps for how to properly wash your hands."
        ],
        "de": [
            "Verfassen Sie eine Reihe einfacher, schrittweise Anweisungen zum Schnüren einer Schuh Schnürsenkel.",
            "Erstellen Sie eine kurze Anleitung, wie man eine perfekte Tasse Kaffee brüht.",
            "Generieren Sie Anweisungen, wie man einen platten Reifen an einem Auto wechselt.",
            "Schreiben Sie eine einfache, klare Anleitung für ein Brettspiel.",
            "Erstellen Sie eine Liste von Schritten für das ordnungsgemäße Händewaschen."
        ]
    },
    "news_reporting": {
        "en": [
            "Write the opening paragraph of a news report about a local election.",
            "Summarize the key facts of a fictional event in the style of a news report.",
            "Generate a short news brief about a recent scientific discovery.",
            "Write a neutral, objective report on a controversial topic.",
            "Create the lead sentence for a breaking news story about a natural disaster."
        ],
        "de": [
            "Schreiben Sie den Eröffnungsabsatz eines Nachrichtenberichts über eine lokale Wahl.",
            "Fassen Sie die wesentlichen Fakten eines fiktiven Ereignisses im Stil eines Nachrichtenberichts zusammen.",
            "Erstellen Sie eine kurze Nachrichtenmitteilung über eine kürzliche wissenschaftliche Entdeckung.",
            "Verfassen Sie einen neutralen, objektiven Bericht über ein umstrittenes Thema.",
            "Erstellen Sie den Leitsatz für eine Nachrichtengeschichte über eine Naturkatastrophe."
        ]
    },
    "scientific_writing": {
        "en": [
            "Write the abstract for a fictional scientific paper on the effects of caffeine on memory.",
            "Describe the methodology for an experiment to test a hypothesis.",
            "Summarize the results of a fictional clinical trial.",
            "Write a brief conclusion for a scientific study.",
            "Formulate a research question for a new scientific investigation."
        ],
        "de": [
            "Verfassen Sie das Abstract für eine fiktive wissenschaftliche Abhandlung über die Wirkung von Koffein auf das Gedächtnis.",
            "Beschreiben Sie die Methodik für ein Experiment, um eine Hypothese zu testen.",
            "Fassen Sie die Ergebnisse einer fiktiven klinischen Studie zusammen.",
            "Verfassen Sie einen kurzen Schlussfolgerung für eine wissenschaftliche Studie.",
            "Formulieren Sie eine Forschungsfrage für eine neue wissenschaftliche Untersuchung."
        ]
    }
}
