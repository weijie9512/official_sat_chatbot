# CHANGED_HERE
import nltk
from regex import S

from model.models import UserModelSession, Choice, UserModelRun, Protocol, UserCommitment
from model.classifiers import get_emotion, get_sentence_score, empathy_score
import pandas as pd
import numpy as np
import random
from collections import deque
import re
import datetime
import time
import os
ROOT_DIR = os.path.abspath(os.path.dirname(__file__)).split("/")
ROOT_DIR = "/".join(ROOT_DIR[:-1])
#CONFIG_PATH = os.path.join(ROOT_DIR, 'configuration.conf')
print(ROOT_DIR)
nltk.download("wordnet")
from nltk.corpus import wordnet  # noqa



COI_IP_ADDRESS = "http://3.11.224.175"
class ModelDecisionMaker:
    def __init__(self):
        #self.kai = pd.read_csv('/Users/weijiechua/Desktop/ImperialClasses/Courses/Term3/wj_SATbot2.0/model/kai.csv', encoding='ISO-8859-1') #change path
        #self.robert = pd.read_csv('/Users/weijiechua/Desktop/ImperialClasses/Courses/Term3/wj_SATbot2.0/model/robert.csv', encoding='ISO-8859-1')
        #self.gabrielle = pd.read_csv('/Users/weijiechua/Desktop/ImperialClasses/Courses/Term3/wj_SATbot2.0/model/gabrielle.csv', encoding='ISO-8859-1')
        #self.arman = pd.read_csv('/Users/weijiechua/Desktop/ImperialClasses/Courses/Term3/wj_SATbot2.0/model/arman.csv', encoding='ISO-8859-1')
        #self.olivia = pd.read_csv('/Users/weijiechua/Desktop/ImperialClasses/Courses/Term3/wj_SATbot2.0/model/olivia.csv', encoding='ISO-8859-1')
       
        self.compassion_data = pd.read_excel(f'{ROOT_DIR}/data/survey_data/combined_data/sat_data_combined.xlsx')
        self.question_bank = pd.read_csv(f'{ROOT_DIR}/data/question_bank.csv', index_col="questioncode")
        
        self.news_war = pd.read_csv(f'{ROOT_DIR}/data/news_data/war.csv')
        self.news_mental = pd.read_csv(f'{ROOT_DIR}/data/news_data/mental.csv')
        self.news_climate = pd.read_csv(f'{ROOT_DIR}/data/news_data/climate.csv')
        self.news_poverty = pd.read_csv(f'/{ROOT_DIR}/data/news_data/poverty.csv')
        self.news_homeless = pd.read_csv(f'{ROOT_DIR}/data/news_data/homeless.csv')
        self.news_wealth_inequality = pd.read_csv(f'{ROOT_DIR}/data/news_data/wealth_inequality.csv')
        self.news_gender_inequality = pd.read_csv(f'{ROOT_DIR}/data/news_data/gender_inequality.csv')

        

        # Titles from workshops (Title 7 adapted to give more information)
        self.PROTOCOL_TITLES = [
            "0: None",
            "1: Connecting with the Child [Week 1]",
            "2: Laughing at our Two Childhood Pictures [Week 1]",
            "3: Falling in Love with the Child [Week 2]",
            "4: Vow to Adopt the Child as Your Own Child [Week 2]",
            "5: Maintaining a Loving Relationship with the Child [Week 3]",
            "6: An exercise to Process the Painful Childhood Events [Week 3]",
            "7: Protocols for Creating Zest for Life [Week 4]",
            "8: Loosening Facial and Body Muscles [Week 4]",
            "9: Protocols for Attachment and Love of Nature  [Week 4]",
            "10: Laughing at, and with One's Self [Week 5]",
            "11: Processing Current Negative Emotions [Week 5]",
            "12: Continuous Laughter [Week 6]",
            "13: Changing Our Perspective for Getting Over Negative Emotions [Week 6]",  # noqa
            "14: Protocols for Socializing the Child [Week 6]",
            "15: Recognising and Controlling Narcissism and the Internal Persecutor [Week 7]",  # noqa
            "16: Creating an Optimal Inner Model [Week 7]",
            "17: Solving Personal Crises [Week 7]",
            "18: Laughing at the Harmless Contradiction of Deep-Rooted Beliefs/Laughing at Trauma [Week 8]",  # noqa
            "19: Changing Ideological Frameworks for Creativity [Week 8]",
            "20: Affirmations [Week 8]",
        ]

        self.TITLE_TO_PROTOCOL = {
            self.PROTOCOL_TITLES[i]: i for i in range(len(self.PROTOCOL_TITLES))
        }

        self.recent_protocols = deque(maxlen=20)
        self.reordered_protocol_questions = {}
        self.protocols_to_suggest = []

        # Goes from user id to actual value
        self.current_run_ids = {}
        self.current_protocol_ids = {}

        self.current_protocols = {}

        self.positive_protocols = [i for i in range(1, 21)]

        self.INTERNAL_PERSECUTOR_PROTOCOLS = [
            self.PROTOCOL_TITLES[15],
            self.PROTOCOL_TITLES[16],
            self.PROTOCOL_TITLES[8],
            self.PROTOCOL_TITLES[19],
        ]

        # Keys: user ids, values: dictionaries describing each choice (in list)
        # and current choice
        self.user_choices = {}

        # Keys: user ids, values: scores for each question
        #self.user_scores = {}

        # Keys: user ids, values: current suggested protocols
        self.suggestions = {}

        # Tracks current emotion of each user after they classify it
        self.user_emotions = {}

        self.guess_emotion_predictions = {}
        # Structure of dictionary: {question: {
        #                           model_prompt: str or list[str],
        #                           choices: {maps user response to next protocol},
        #                           protocols: {maps user response to protocols to suggest},
        #                           }, ...
        #                           }
        # This could be adapted to be part of a JSON file (would need to address
        # mapping callable functions over for parsing).

        self.users_names = {}
        self.remaining_choices = {}

        self.recent_questions = {}

        self.chosen_personas = {}
        self.datasets = {}


        self.user_free_text = {}
        self.compassion_energy_weightage = {"esa_consistency": 2, "auc_understanding_consistency": 3, "auc_commitments_consistency": 3, "empathetic_score": 2}
        self.nodes_count_by_user = {}
        self.nodes_direction = {}
        self.news_by_category = {"mental": self.news_mental, "war": self.news_war, "climate": self.news_climate, "poverty": self.news_poverty,  \
                                "homeless": self.news_homeless, "wealth inequality": self.news_wealth_inequality, "gender inequality": self.news_gender_inequality}
        
        self.no_lowercase_node = ["greet_user2", "sat_ask_why_not_help_vulnerable_communities", "auc_awareness_no_feeling", \
                                "esa_last_week", "coi_start", "coi_start2a", "coi_start2b","sat_ask_type_of_compassion",
                                "auc_awareness_begin", "auc_awareness_get_news", "auc_awareness_feel_reading_news", \
                                "auc_awareness_quick_test", "auc_commitments_why_not_follow_through_solutions", "esa_show_statistics", "check_tender_vs_foresighted_compassion"]
        
        """
        READ HERE
        # choices["open_text"] means that what is the next prompt to return. 
        For example, in "ask_name", self.save_name(user_id) return "choose_persona" which determines next choice

        """
        self.QUESTIONS = {

            "ask_name": {
               "model_prompt": "Please enter your first name:",
               "choices": {
                   "open_text": lambda user_id, db_session, curr_session, app: self.save_name(user_id)
               },
               "protocols": {"open_text": []},
           },


           "choose_persona": {
              "model_prompt": "Who would you like to talk to?",
              "choices": {
                  "Kai": lambda user_id, db_session, curr_session, app: self.get_kai(user_id),
                  "Robert": lambda user_id, db_session, curr_session, app: self.get_robert(user_id),
                  "Gabrielle": lambda user_id, db_session, curr_session, app: self.get_gabrielle(user_id),
                  "Arman": lambda user_id, db_session, curr_session, app: self.get_arman(user_id),
                  "Olivia": lambda user_id, db_session, curr_session, app: self.get_olivia(user_id),
              },
              "protocols": {
                  "Kai": [],
                  "Robert": [],
                  "Gabrielle": [],
                  "Arman": [],
                  "Olivia": [],
              },
          },

            "greet_user": { 
                "model_prompt": lambda user_id, db_session, curr_session, app: self.greet_user(user_id),

                "choices": {
                # CHANGED_HERE/ CHANGED/ REVERT
                    "Okay, but what is the goal?": "greet_user2"
                #    "Continue": "main_node",
                },

                "protocols": {
                   "Okay, but what is the goal?": [],
               },

            },

            "greet_user2": { 
                "model_prompt": lambda user_id, db_session, curr_session, app: self.greet_user2(user_id),

                "choices": {
                    "Continue": "opening_prompt",
                    "I am very familiar, bring me to the MAIN NODE": "main_node",
                },

                "protocols": {
                   "Continue": [],
                   "I am very familiar, bring me to the MAIN NODE": [],
               },

            },


            "opening_prompt": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_opening_prompt(user_id),

                "choices": {
                    "open_text": lambda user_id, db_session, curr_session, app: self.determine_next_prompt_opening(user_id, app, db_session)
                },
                "protocols": {"open_text": []},
            },

            "guess_emotion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_guess_emotion(
                    user_id, app, db_session
                ),
                "choices": {
                    "yes": {
                        "Sad": "trying_protocol_9",
                        "Angry": "trying_protocol_9",
                        "Anxious/Scared": "trying_protocol_9",
                        "Happy/Content": "transfer_before_main_node",
                    },
                    "no": "check_emotion",
                },
                "protocols": {
                    "yes": [],
                    "no": []
                    },
            },


            "check_emotion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_check_emotion(user_id, app, db_session),

                "choices": {
                    "Sad": "trying_protocol_9",
                    #"Angry": lambda user_id, db_session, curr_session, app: self.get_angry_emotion(user_id),
                    #"Anxious/Scared": lambda user_id, db_session, curr_session, app: self.get_anxious_emotion(user_id),
                    "Happy/Content": "transfer_before_main_node",
                },
                "protocols": {
                    "Sad": [],
                    #"Angry": [],
                    #"Anxious/Scared" : [],
                    "Happy/Content": []
                },
            },

            ############# NEGATIVE EMOTIONS (SADNESS, ANGER, FEAR/ANXIETY)


            "after_classification_negative": {
                #"model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_specific_event(user_id, app, db_session),
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_overcome_negative_emotions(user_id, app, db_session),
                
                "choices": {
                    self.PROTOCOL_TITLES[13]: "trying_protocol_9",
                },

                "protocols": {
                    self.PROTOCOL_TITLES[13]: [],

                },
            },

            

            "event_is_recent": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_event_is_recent(user_id, app, db_session),

                "choices": {
                    "It was recent": "revisiting_recent_events",
                    "It was distant": "revisiting_distant_events",
                },
                "protocols": {
                    "It was recent": [],
                    "It was distant": []
                    },
            },

            "revisiting_recent_events": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_revisit_recent(user_id, app, db_session),

                "choices": {
                    "yes": "more_questions",
                    "no": "more_questions",
                },
                "protocols": {
                    "yes": [self.PROTOCOL_TITLES[7], self.PROTOCOL_TITLES[8]],
                    "no": [self.PROTOCOL_TITLES[11]],
                },
            },

            "revisiting_distant_events": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_revisit_distant(user_id, app, db_session),

                "choices": {
                    "yes": "more_questions",
                    "no": "more_questions",
                },
                "protocols": {
                    "yes": [self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[17]],
                    "no": [self.PROTOCOL_TITLES[6]]
                },
            },

            "more_questions": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_more_questions(user_id, app, db_session),

                "choices": {
                    "Okay": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
                    "I'd rather not": "project_emotion",
                },
                "protocols": {
                    "Okay": [],
                    "I'd rather not": [self.PROTOCOL_TITLES[13]],
                },
            },

            "displaying_antisocial_behaviour": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_antisocial(user_id, app, db_session),

                "choices": {
                    "yes": "project_emotion",
                    "no": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
                },
                "protocols": {
                    "yes": [self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[14]],
                    "no": [self.PROTOCOL_TITLES[13]],
                },
            },

            "internal_persecutor_saviour": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_saviour(user_id, app, db_session),

                "choices": {
                    "yes": "project_emotion",
                    "no": "internal_persecutor_victim",
                },
                "protocols": {
                    "yes": self.INTERNAL_PERSECUTOR_PROTOCOLS,
                    "no": []
                },
            },

            "internal_persecutor_victim": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_victim(user_id, app, db_session),

                "choices": {
                    "yes": "project_emotion",
                    "no": "internal_persecutor_controlling",
                },
                "protocols": {
                    "yes": self.INTERNAL_PERSECUTOR_PROTOCOLS,
                    "no": []
                },
            },

            "internal_persecutor_controlling": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_controlling(user_id, app, db_session),

                "choices": {
                "yes": "project_emotion",
                "no": "internal_persecutor_accusing"
                },
                "protocols": {
                "yes": self.INTERNAL_PERSECUTOR_PROTOCOLS,
                "no": []
                },
            },

            "internal_persecutor_accusing": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_accusing(user_id, app, db_session),

                "choices": {
                "yes": "project_emotion",
                "no": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
                },
                "protocols": {
                "yes": self.INTERNAL_PERSECUTOR_PROTOCOLS,
                "no": [self.PROTOCOL_TITLES[13]],
                },
            },

            "rigid_thought": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_rigid_thought(user_id, app, db_session),

                "choices": {
                    "yes": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
                    "no": "project_emotion",
                },
                "protocols": {
                    "yes": [self.PROTOCOL_TITLES[13]],
                    "no": [self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[19]],
                },
            },


            "personal_crisis": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_personal_crisis(user_id, app, db_session),

                "choices": {
                    "yes": "project_emotion",
                    "no": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
                },
                "protocols": {
                    "yes": [self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[17]],
                    "no": [self.PROTOCOL_TITLES[13]],
                },
            },

            ################# POSITIVE EMOTION (HAPPINESS/CONTENT) #################

            "after_classification_positive": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_happy(user_id, app, db_session),

                "choices": {
                    "continue": "main_node",
                },
                "protocols": {
                    "continue": [],
                },
            },

            ############################# ALL EMOTIONS #############################

            "project_emotion": {
               "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_project_emotion(user_id, app, db_session),

               "choices": {
                   "Continue": "sat_feel_compassion_to_child_question",
               },
               "protocols": {
                   "Continue": [],
               },
            },


            "suggestions": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggestions(user_id, app, db_session),

                "choices": {
                     self.PROTOCOL_TITLES[k]: "trying_protocol" #self.current_protocol_ids[user_id]
                     for k in self.positive_protocols
                },
                "protocols": {
                     self.PROTOCOL_TITLES[k]: [self.PROTOCOL_TITLES[k]]
                     for k in self.positive_protocols
                },
            },

            
            # back to beginning

            "trying_protocol": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_trying_protocol(user_id, app, db_session),

                "choices": {"continue": "user_found_useful"},
                "protocols": {"continue": []},
            },

            

            "user_found_useful": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_found_useful(user_id, app, db_session),

                "choices": {
                    "I feel better": "new_protocol_better",
                    "I feel worse": "new_protocol_worse",
                    "I feel no change": "new_protocol_same",
                },
                "protocols": {
                    "I feel better": [],
                    "I feel worse": [],
                    "I feel no change": []
                },
            },

            "new_protocol_better": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_new_better(user_id, app, db_session),

                "choices": {
                    "Yes (show follow-up suggestions)": lambda user_id, db_session, curr_session, app: self.determine_next_prompt_new_protocol(
                        user_id, app
                    ),
                    "Yes (restart questions)": "restart_prompt",
                    "No (end session)": "ending_prompt",
                },
                "protocols": {
                    "Yes (show follow-up suggestions)": [],
                    "Yes (restart questions)": [],
                    "No (end session)": []
                },
            },

            "new_protocol_worse": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_new_worse(user_id, app, db_session),

                "choices": {
                    "Yes (show follow-up suggestions)": lambda user_id, db_session, curr_session, app: self.determine_next_prompt_new_protocol(
                        user_id, app
                    ),
                    "Yes (restart questions)": "restart_prompt",
                    "No (end session)": "ending_prompt",
                },
                "protocols": {
                    "Yes (show follow-up suggestions)": [],
                    "Yes (restart questions)": [],
                    "No (end session)": []
                },
            },

            "new_protocol_same": {
                "model_prompt": [
                                "I am sorry to hear you have not detected any change in your mood.",
                                "That can sometimes happen but if you agree we could try another protocol and see if that is more helpful to you.",
                                "Would you like me to suggest a different protocol?"
                                ],

                "choices": {
                    "Yes (show follow-up suggestions)": lambda user_id, db_session, curr_session, app: self.determine_next_prompt_new_protocol(
                        user_id, app
                    ),
                    "Yes (restart questions)": "restart_prompt",
                    "No (end session)": "ending_prompt",
                },
                "protocols": {
                    "Yes (show follow-up suggestions)": [],
                    "Yes (restart questions)": [],
                    "No (end session)": []
                },
            },

            "ending_prompt": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_ending(user_id, app, db_session),

                "choices": {"any": "opening_prompt"},
                "protocols": {"any": []}
                },

            "restart_prompt": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_restart_prompt(user_id),

                "choices": {
                    "open_text": lambda user_id, db_session, curr_session, app: self.determine_next_prompt_opening(user_id, app, db_session)
                },
                "protocols": {"open_text": []},
            },

              ############################### SHORTCUT: COMPASSION BEFORE MAIN NODE ############################

            "trying_protocol_9": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.trying_protocol_9(user_id, app, db_session),

                "choices": {"continue": "user_found_compassion_to_child"},
                "protocols": {"continue": []},
            },


            "user_found_compassion_to_child": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_found_compassion_to_child(user_id, app, db_session),

                "choices": {
                    "yes": "check_tender_vs_foresighted_compassion",
                    "no": lambda user_id, db_session, curr_session, app: self.get_count_by_question(user_id, curr_session, "A05"),
                    #"no": "trying_protocol_9",
                },
                "protocols": {
                    "yes": [],
                    "no": [],
                },
            },

            "user_found_no_compassion_to_child_after_3": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.user_found_no_compassion_to_child_after_3(user_id, app, db_session),

                "choices": {
                    "continue": "main_node",
                },
                "protocols": {
                    "continue": [],
                },
            },

    

            "check_tender_vs_foresighted_compassion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.check_tender_vs_foresighted_compassion(user_id, app, db_session),
                "choices": {
                    "Protect": "shown_tender_compassion",
                    "Grow up": "shown_foresighted_compassion"},


                "protocols": {
                    "Protect": [],
                    "Grow up": [],
                },
            },

            "shown_tender_compassion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.shown_tender_compassion(user_id, app, db_session),
                "choices": {
                    "continue": "main_node",
                },


                "protocols": {
                    "continue": [],
                },
            },

            "shown_foresighted_compassion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.shown_foresighted_compassion(user_id, app, db_session),
                "choices": {
                    "continue": "main_node",
                },


                "protocols": {
                    "continue": [],
                },
            },

            "shown_compassion_ask_feel_better":{
                 "model_prompt": lambda user_id, db_session, curr_session, app: self.shown_compassion_ask_feel_better(user_id, app, db_session),
                "choices": {
                    "yes": "main_node",
                    "no": "trying_protocol_9",
                },


                "protocols": {
                    "yes": [],
                    "no": [],
                },

            },


            ############################### SHORTCUT: MAIN NODE ############################
            "main_node": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.main_node(user_id, app, db_session),
                "choices": {
                    "What should I choose?": "main_node_faq",
                    "Awareness, Understanding, Commitments (AUC)": "auc_choose_a_u_c",
                    "Clash of Ideas (COI)": "coi_start",
                    "Everyday Simple Action (ESA)": "esa_last_week",
                    "SAT protocols (SAT)": "sat_compassion_start", 
                    "End the session": "ending_prompt",
                },

                "protocols": {
                    "What should I choose?": [],
                    "Awareness, Understanding, Commitments (AUC)": [],
                    "Clash of Ideas (COI)": [],
                    "Everyday Simple Action (ESA)": [],
                    "SAT protocols (SAT)": [], 
                    "End the session": [],
                },
            },

            "main_node_faq": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.main_node_faq(user_id, app, db_session),
                "choices": {
                    "continue": "main_node",
                },

                "protocols": {
                    "continue": [],
                },
            },

            "congrats_before_main_node": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.congrats_before_main_node(user_id, app, db_session),
                "choices": {
                    "continue": "main_node",
                },

                "protocols": {
                    "continue": [],
                },
            },

            "transfer_before_main_node": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.transfer_before_main_node(user_id, app, db_session),
                "choices": {
                    "continue": "main_node",
                },

                "protocols": {
                    "continue": [],
                },
            },

            "come_back_tomorrow_before_main_node": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.come_back_tomorrow_before_main_node(user_id, app, db_session),
                "choices": {
                    "continue": "main_node",
                },

                "protocols": {
                    "continue": [],
                },
            },



            ############################### SHORTCUT: coi ############################
            "coi_start": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.coi_start(user_id, app, db_session),
                "choices": {
                    "Show me the template and link!": "coi_start2a",
                    "Show me the link!": "coi_start2b",
                },

                "protocols": {
                    "Show me the template and link!": [],
                    "Show me the link!": [],
                },
            },

            "coi_start2a": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.coi_start2a(user_id, app, db_session),
                "choices": {
                    "I'm inspired, bring me back to MAIN NODE!": "transfer_before_main_node",
                },

                "protocols": {
                    "I'm inspired, bring me back to MAIN NODE!": [],
                },
            },

            "coi_start2b": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.coi_start2b(user_id, app, db_session),
                "choices": {
                    "I'm inspired, bring me back to MAIN NODE!": "transfer_before_main_node",
                },

                "protocols": {
                    "I'm inspired, bring me back to MAIN NODE!": [],
                },
            },


            ############################### SHORTCUT: Everyday simple action (ESA) ############################
            "esa_last_week": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.esa_last_week(user_id, app, db_session),
                "choices": {
                    "Yes, I feel great after following through the actions last week!": lambda user_id, db_session, curr_session, app: self.esa_set_count(user_id, db_session, "yes"),
                    "No, unfortunately I couldn't follow through actions last week.": lambda user_id, db_session, curr_session, app: self.esa_set_count(user_id, db_session, "no"),
                    "This is my first time doing it!": "esa_simple_scenario",
                },

                "protocols": {
                    "Yes, I feel great after following through the actions last week!": [],
                    "No, unfortunately I couldn't follow through actions last week.": [],
                    "This is my first time doing it!": [],
                },
            },

            "esa_show_statistics": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.esa_show_statistics(user_id),
                "choices": {
                    "Got it!": "esa_simple_scenario",
                },

                "protocols": {
                    "Got it!": [],
                },
            },

            "esa_simple_scenario": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.esa_simple_scenario(user_id),
                "choices": {
                    "I will do my best": "transfer_before_main_node",
                },

                "protocols": {
                    "I will do my best": [],
                },
            },


            ############################### SHORTCUT: SAT PROTOCOLS (SAT) ############################
            "sat_compassion_start": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_compassion_start(user_id, app, db_session),
                "choices": {
                    "continue": "sat_compassion_difference",
                },

                "protocols": {
                    "continue": [],
                },

            },

            "sat_compassion_difference": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_compassion_difference(user_id, app, db_session),
                "choices": {
                    "yes": "sat_compassion_difference_example",
                    "no": "project_emotion_to_child",
                },
                "protocols": {
                    "yes": [],
                    "no": []
                },

            },

            "sat_compassion_difference_example": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_compassion_difference_example(user_id, app, db_session),
                "choices": {
                    "continue": "project_emotion_to_child",
                },
                "protocols": {
                    "continue": [],
                },

            },

            "project_emotion_to_child": {
               "model_prompt": lambda user_id, db_session, curr_session, app: self.project_emotion_to_child(user_id, app, db_session),

               "choices": {
                   "continue": "sat_feel_compassion_to_child_question",
               },
               "protocols": {
                   "continue": [],
               },
            },

            "sat_feel_compassion_to_child_question":{
                 "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_feel_compassion_to_child_question(user_id, app, db_session),

               "choices": {
                   "yes": "sat_ask_type_of_compassion", # PART 3
                   "no": "sat_part1_begin", # PART 1
               },
               "protocols": {
                   "yes": [],
                   "no": [],
               },

            },

             ############################### SHORTCUT: SAT PROTOCOLS (SAT) PART 1 ############################

            "sat_part1_begin":{
                 "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_part1_begin(user_id, app, db_session),

               "choices": {
                   "continue": "trying_protocol_1_and_2",
               },
               "protocols": {
                  "continue": [],
               },

            },

            "trying_protocol_1_and_2":{
                 "model_prompt": lambda user_id, db_session, curr_session, app: self.trying_protocol_1_and_2(user_id, app, db_session),

               "choices": {
                   "continue": "sat_form_connection_with_child_level1",
               },
               "protocols": {
                  "continue": [],
               },

            },

            "sat_form_connection_with_child_level1":{
                 "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_form_connection_with_child_level1(user_id, app, db_session),

               "choices": {
                   "yes": "sat_take_child_relationship_to_next_level",
                   "no": "trying_protocol_1_and_2"
               },
               "protocols": {
                  "yes": [],
                   "no": [],
               },

            },
            "sat_take_child_relationship_to_next_level":{
                 "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_take_child_relationship_to_next_level(user_id, app, db_session),

               "choices": {
                   "yes": "trying_protocol_3_and_4_and_5",
                   "no": "sat_go_esa_or_end"
               },
               "protocols": {
                  "yes": [],
                   "no": [],
               },

            },

            "sat_go_esa_or_end":{
                 "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_go_esa_or_end(user_id, app, db_session),

               "choices": {
                   "ESA": "transfer_to_main_node",
                   "End": "ending_prompt"
               },
               "protocols": {
                  "ESA": [],
                   "End": [],
               },

            },

            "trying_protocol_3_and_4_and_5":{
                 "model_prompt": lambda user_id, db_session, curr_session, app: self.trying_protocol_3_and_4_and_5(user_id, app, db_session),

               "choices": {
                   "continue": "sat_form_connection_with_child_level2",
               },
               "protocols": {
                  "continue": [],
               },

            },

            "sat_form_connection_with_child_level2":{
                 "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_form_connection_with_child_level2(user_id, app, db_session),

               "choices": {
                   "yes": "sat_completed_tender_compassion_training",
                   "no": "trying_protocol_3_and_4_and_5"
               },
               "protocols": {
                  "yes": [],
                   "no": [],
               },

            },

            "sat_completed_tender_compassion_training":{ 
                "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_completed_tender_compassion_training(user_id, app, db_session),

               "choices": {
                  "continue": "sat_shown_tender_compassion",
               },
               "protocols": {
                  "continue": [],
               },

            },

            
            ############################### SHORTCUT: SAT PROTOCOLS (SAT) PART 2 ############################
            "sat_ask_type_of_compassion":{
                 "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_ask_type_of_compassion(user_id, app, db_session),

               "choices": {
                   "Tender Compassion": "sat_shown_tender_compassion",
                   "Foresighted Compassion": "sat_ask_user_if_want_protocol_8",
               },
               "protocols": {
                   "Tender Compassion": [],
                   "Foresighted Compassion": [],
               },
            },

            
            "sat_shown_tender_compassion":{
                "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_shown_tender_compassion(user_id, app, db_session),

               "choices": {
                  "continue": "sat_ask_if_help_vulnerable_communities",
               },
               "protocols": {
                  "continue": [],
               },

            },

            "sat_ask_if_help_vulnerable_communities":{
                "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_ask_if_help_vulnerable_communities(user_id, app, db_session),

               "choices": {
                  "yes": "sat_how_to_help_vulnerable_community",
                  "no": "sat_ask_why_not_help_vulnerable_communities",
               },
               "protocols": {
                  "yes": [],
                  "no": [],
               },

            },

            "sat_ask_why_not_help_vulnerable_communities":{
                "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_ask_why_not_help_vulnerable_communities(user_id, app, db_session),

               "choices": {
                  "Unable to feel compassion": "sat_answer_cant_feel_compassion",
                  "Do not deserve help": "trying_protocol_15",
               },
               "protocols": {
                  "Unable to feel compassion": [],
                  "Do not deserve help": [],
               },

            },

            "sat_answer_cant_feel_compassion":{
                "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_answer_cant_feel_compassion(user_id, app, db_session),

               "choices": {
                  "continue": "trying_protocol_1_and_2",
               },
               "protocols": {
                  "continue": [],
               },

            },

            "trying_protocol_15": {
                 "model_prompt": lambda user_id, db_session, curr_session, app: self.trying_protocol_15(user_id, app, db_session),

               "choices": {
                  "continue": "sat_imagine_self_vulnerable",
               },
               "protocols": {
                  "continue": [],
               },

            },


            "sat_imagine_self_vulnerable": {
                 "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_imagine_self_vulnerable(user_id, app, db_session),

               "choices": {
                   "yes": "sat_ask_if_help_vulnerable_communities2", 
                  "no": "sat_vulnerable_communities_think_deeper", # part 4
               },
               "protocols": {
                  "yes": [],
                  "no": [],
               },

            },

            "sat_ask_if_help_vulnerable_communities2":{
                "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_ask_if_help_vulnerable_communities2(user_id, app, db_session),

               "choices": {
                  "yes": "sat_how_to_help_vulnerable_community", # part 4
                  "no": "trying_protocol_15",
               },
               "protocols": {
                  "yes": [],
                  "no": [],
               },

            },
            "sat_vulnerable_communities_think_deeper":{
                "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_vulnerable_communities_think_deeper(user_id, app, db_session),

               "choices": {
                  "yes": "sat_how_to_help_vulnerable_community", # part 4
                  "no": lambda user_id, db_session, curr_session, app: self.get_count_by_question(user_id, curr_session, "C09"),
               },
               "protocols": {
                  "yes": [],
                  "no": [],
               },


            },


        




            ############################### SHORTCUT: SAT PROTOCOLS (SAT) PART 3 ############################

            

            "sat_ask_user_if_want_protocol_8":{
                 "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_ask_user_if_want_protocol_8(user_id, app, db_session),

               "choices": {
                   "yes": "trying_protocol_8",
                   "no": "congrats_before_main_node",
               },
               "protocols": {
                   "yes": [],
                   "no": [],
               },
            },

            "trying_protocol_8": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.trying_protocol_8(user_id, app, db_session),

                "choices": {
                    "continue": "congrats_before_main_node"
                },

                "protocols": {
                    "continue": []
                },
            },

            ############################### SHORTCUT: SAT PROTOCOLS (SAT) PART 4 ############################
            "sat_how_to_help_vulnerable_community":{
                "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_how_to_help_vulnerable_community(user_id, app, db_session),

               "choices": {
                   "Give them what they want": "trying_protocol_17_and_18",
                   "Research what is best for them, and give them what they need": "sat_shown_foresighted_compassion_and_ready",
               },
               "protocols": {
                   "Give them what they want": [],
                   "Research what is best for them, and give them what they need": [],
               },

            },

            "sat_shown_foresighted_compassion_and_ready":{
                "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_shown_foresighted_compassion_and_ready(user_id, app, db_session),

               "choices": {
                   "continue": "main_node",
               },
               "protocols": {
                   "continue": [],
               },

            },

            "trying_protocol_17_and_18":{
                "model_prompt": lambda user_id, db_session, curr_session, app: self.trying_protocol_17_and_18(user_id, app, db_session),

               "choices": {
                   "continue": "sat_immediate_action_or_maturity",
               },
               "protocols": {
                   "continue": [],
               },

            },

            "sat_immediate_action_or_maturity": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_immediate_action_or_maturity(user_id, app, db_session),

               "choices": {
                   "yes": "sat_completed_foresighted_compassion_training",
                   "no" : "trying_protocol_17_and_18",
               },
               "protocols": {
                   "yes": [],
                   "no": [],
               },

            },

            "sat_completed_foresighted_compassion_training":{
                "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_completed_foresighted_compassion_training(user_id, app, db_session),

               "choices": {
                   "continue": "main_node",
               },
               "protocols": {
                   "continue": [],
               },


            },






            ############################### SHORTCUT: SAT PROTOCOLS (SAT) PART 5 ############################

            "sat_might_be_anti_social":{
                 "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_might_be_anti_social(user_id, app, db_session),

               "choices": {
                  "continue": "sat_personal_resentment",
               },
               "protocols": {
                  "continue": [],
               },
                
            },

            
            "sat_personal_resentment": {
                 "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_personal_resentment(user_id, app, db_session),

               "choices": {
                  "yes": "trying_protocol_17_and_18_anti_social",
                  "no": "sat_care_about_other_people",
               },
               "protocols": {
                  "yes": [],
                  "no": [],
               },
            },

            "trying_protocol_17_and_18_anti_social": {
                 "model_prompt": lambda user_id, db_session, curr_session, app: self.trying_protocol_17_and_18_anti_social(user_id, app, db_session),

               "choices": {
                  "continue": "sat_ask_care_social",
               },
               "protocols": {
                  "continue": [],
               },

            },

            "sat_ask_care_social":{
                "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_ask_care_social(user_id, app, db_session),

               "choices": {
                  "yes": "congrats_before_main_node",
                  "no": "trying_protocol_17_and_18_anti_social",
               },
               "protocols": {
                  "yes": [],
                  "no": [],
               },   

            },


            "sat_care_about_other_people": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_care_about_other_people(user_id, app, db_session),

               "choices": {
                  "yes": "sat_local_community",
                  "no": "trying_protocol_17_and_18_anti_social",
               },
               "protocols": {
                  "yes": [],
                  "no": [],

               },
            
            },
            "sat_local_community": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_local_community(user_id, app, db_session),

               "choices": {
                  "yes": "sat_global_community",
                  "no": "trying_protocol_17_and_18_anti_social",
               },
               "protocols": {
                  "yes": [],
                  "no": [],

               },
            
            },

            "sat_global_community": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_global_community(user_id, app, db_session),

               "choices": {
                  "yes": "sat_global_community_compassion",
                  "no": "trying_protocol_17_and_18_anti_social",
               },
               "protocols": {
                  "yes": [],
                  "no": [],

               },
            
            },

            "sat_global_community_compassion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_global_community_compassion(user_id, app, db_session),

               "choices": {
                  "yes": "sat_no_anti_social",
                  "no": "trying_protocol_17_and_18_anti_social",
               },
               "protocols": {
                  "yes": [],
                  "no": [],
               },
            
            },

            "sat_no_anti_social": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.sat_no_anti_social(user_id, app, db_session),

               "choices": {
                  "continue": "sat_how_to_help_vulnerable_community",
               },
               "protocols": {
                  "continue": [],
               },
            
            },

            ############################### SHORTCUT: AUC ############################
            "auc_choose_a_u_c": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_choose_a_u_c(user_id, app, db_session),

               "choices": {
                  "Awareness": "auc_awareness_begin",
                  "Understanding": "auc_understanding_begin",
                  "Commitments": "auc_commitments_begin",
               },
               "protocols": {
                  "Awareness": [],
                  "Understanding": [],
                  "Commitments": [],
               },

            
            },

            # SHORTCUT: AUC AWARENESS
            "auc_awareness_begin": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_awareness_begin(user_id, app, db_session),
                
               "choices": {
                  "Mental Health": lambda user_id, db_session, curr_session, app: self.auc_awareness_set_news(user_id, app, db_session, "mental"),
                  "War": lambda user_id, db_session, curr_session, app: self.auc_awareness_set_news(user_id, app, db_session, "war"),
                  "Climate": lambda user_id, db_session, curr_session, app: self.auc_awareness_set_news(user_id, app, db_session, "climate"),
                  "Homeless": lambda user_id, db_session, curr_session, app: self.auc_awareness_set_news(user_id, app, db_session, "homeless"),
                  "Poverty": lambda user_id, db_session, curr_session, app: self.auc_awareness_set_news(user_id, app, db_session, "poverty"),
                  "Gender Inequality": lambda user_id, db_session, curr_session, app: self.auc_awareness_set_news(user_id, app, db_session, "gender inequality"),
                  "Wealth Inequality": lambda user_id, db_session, curr_session, app: self.auc_awareness_set_news(user_id, app, db_session, "wealth inequality"),
               },
               "protocols": {
                  "Mental Health": [],
                  "War": [],
                  "Climate": [],
                  "Homeless": [],
                  "Poverty": [],
                  "Gender Inequality": [],
                  "Wealth Inequality": [],
               },

            
            },
            "auc_awareness_get_news": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_awareness_get_news(user_id, app, db_session),

               "choices": {
                  "Continue": "auc_awareness_feel_reading_news",
                  'Another news': "auc_awareness_get_news",
                  "Another category": "auc_awareness_begin",
               },
               "protocols": {
                  "Continue": [],
                  "Another news": [],
                  "Another category": [],
               },
            
            },


            "auc_awareness_feel_reading_news": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_awareness_feel_reading_news(user_id, app, db_session),

               "choices": {
                  "No feeling": "auc_awareness_no_feeling",
                  "Feel the compassion energy": "auc_awareness_feel_compassion",
               },
               "protocols": {
                  "No feeling": [],
                  "Feel the compassion energy": [],
               },
            
            },

            "auc_awareness_get_another_news": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_awareness_get_another_news(user_id, app, db_session),
                
               "choices": {
                  "Mental": lambda user_id, db_session, curr_session, app: self.auc_awareness_set_news(user_id, app, db_session, "mental"),
                  "War": lambda user_id, db_session, curr_session, app: self.auc_awareness_set_news(user_id, app, db_session, "war"),
                  "Climate": lambda user_id, db_session, curr_session, app: self.auc_awareness_set_news(user_id, app, db_session, "climate"),
                  "Homeless": lambda user_id, db_session, curr_session, app: self.auc_awareness_set_news(user_id, app, db_session, "homeless"),
                  "Poverty": lambda user_id, db_session, curr_session, app: self.auc_awareness_set_news(user_id, app, db_session, "poverty"),
                  "Gender Inequality": lambda user_id, db_session, curr_session, app: self.auc_awareness_set_news(user_id, app, db_session, "gender inequality"),
                  "Wealth Inequality": lambda user_id, db_session, curr_session, app: self.auc_awareness_set_news(user_id, app, db_session, "wealth inequality"),
               },
               "protocols": {
                  "Mental": [],
                  "War": [],
                  "Climate": [],
                  "Homeless": [],
                  "Poverty": [],
                  "Gender Inequality": [],
                  "Wealth Inequality": [],
               },

            
            },

            "auc_awareness_no_feeling": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_awareness_no_feeling(user_id, app, db_session),

               "choices": {
                  "Can't relate, the news feel foreign": "auc_awareness_get_another_news",
                  "Can't feel the compassion energy": "auc_awareness_go_practice_sat",
               },
               "protocols": {
                 "Can't relate, the news feel foreign": [],
                  "Can't feel the compassion energy": [],
               },
            
            },

            "auc_awareness_go_practice_sat": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_awareness_go_practice_sat(user_id, app, db_session),

               "choices": {
                  "continue": "sat_compassion_start",
               },
               "protocols": {
                 "continue": [],
               },
            
            },


            "auc_awareness_feel_compassion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_awareness_feel_compassion(user_id, app, db_session),

               "choices": {
                  "continue": "auc_awareness_quick_test",
               },
               "protocols": {
                 "continue": [],
               },
            
            },

            "auc_awareness_quick_test": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_awareness_quick_test(user_id, app, db_session),

               "choices": {
                  "Reducing own electricity usage": "auc_awareness_explain_compassion_difference",
                  "Reducing own plastic bag usage": "auc_awareness_explain_compassion_difference",
                  "Enforce carbon-neutral policies": "auc_awareness_congrats",
                  "Pressuring government to encourage green-energy related investments": "auc_awareness_congrats",
               },
               "protocols": {
                 "Reducing own electricity usage": [],
                  "Reducing own plastic bag usage": [],
                  "Enforce carbon-neutral policies": [],
                  "Pressuring government to encourage green-energy related investments": [],
               },
            
            },

            "auc_awareness_explain_compassion_difference": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_awareness_explain_compassion_difference(user_id, app, db_session),

               "choices": {
                  "continue": "auc_awareness_congrats",
               },
               "protocols": {
                 "continue": [],
               },
            
            },

            "auc_awareness_congrats": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_awareness_congrats(user_id, app, db_session),

               "choices": {
                  "continue": "transfer_before_main_node",
               },
               "protocols": {
                 "continue": [],
               },
            
            },


            # SHORTCUT: AUC UNDERSTANDING
            "auc_understanding_begin": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_understanding_begin(user_id, app, db_session),

               "choices": {
                  "continue": "auc_understanding_research_existing",
               },
               "protocols": {
                 "continue": [],
               },
            
            },


            "auc_understanding_research_existing": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_understanding_research_existing(user_id, app, db_session),
                
               "choices": {
                   "yes": lambda user_id, db_session, curr_session, app: self.auc_understanding_research_set_count(user_id, db_session, "yes"), # go to auc_understanding_research_statistics
                   "no": lambda user_id, db_session, curr_session, app: self.auc_understanding_research_set_count(user_id, db_session, "no"), # go to auc_understanding_research_statistics
                  #"yes": "auc_understanding_research_reasonable_solutions",
                  # "no": "auc_understanding_set_deadline",
               },
               "protocols": {
                 "yes": [],
                 "no": [],
               },

            },

            "auc_understanding_research_statistics": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_understanding_research_statistics(user_id),

               "choices": {
                   # if yes go to "auc_understanding_research_reasonable_solutions", if not go to "auc_understanding_set_deadline"
                  "continue": lambda user_id, db_session, curr_session, app: self.auc_understanding_research_decide_direction(user_id, db_session), 
               },
               "protocols": {
                 "continue": [],
               },
            
            },

            "auc_understanding_set_deadline": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_understanding_set_deadline(user_id, app, db_session),

               "choices": {
                  "continue": "transfer_before_main_node",
               },
               "protocols": {
                 "continue": [],
               },
            
            },

            "auc_understanding_research_reasonable_solutions": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_understanding_research_reasonable_solutions(user_id, app, db_session),

               "choices": {
                  "yes": "auc_understanding_write_plan",
                   "no": "auc_understanding_go_coi",
               },
               "protocols": {
                 "yes": [],
                 "no": [],
               },

            },

            "auc_understanding_go_coi": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_understanding_go_coi(user_id, app, db_session),

               "choices": {
                  "continue": "transfer_before_main_node",
               },
               "protocols": {
                 "continue": [],
               },

            },

            "auc_understanding_write_plan": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_understanding_write_plan(user_id, app, db_session),

               "choices": {
                  "continue": "transfer_before_main_node",
               },
               "protocols": {
                 "continue": [],
               },

            },

            # SHORTCUT: AUC COMMITMENTS

            "auc_commitments_begin": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_commitments_begin(user_id, app, db_session),

               "choices": {
                  "continue": "auc_commitments_follow_through_solutions",
               },
               "protocols": {
                 "continue": [],
               },
            },


            "auc_commitments_follow_through_solutions": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_commitments_follow_through_solutions(user_id, app, db_session),

               "choices": {
                  "yes": lambda user_id, db_session, curr_session, app: self.auc_commitments_research_set_count(user_id, db_session, "yes"),
                  "no": lambda user_id, db_session, curr_session, app: self.auc_commitments_research_set_count(user_id, db_session, "no"),
               },
               "protocols": {
                  "yes": [],
                  "no": [],
               },
            },

            "auc_commitments_research_statistics": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_commitments_research_statistics(user_id),

               "choices": {
                  "continue": lambda user_id, db_session, curr_session, app: self.auc_commitments_research_decide_direction(user_id, db_session),
               },
               "protocols": {
                 "continue": [],
               },
            },



             "auc_commitments_why_not_follow_through_solutions": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_commitments_why_not_follow_through_solutions(user_id, app, db_session),

               "choices": {
                  "Not practical": "auc_commitments_reflect_on_understanding",
                  "Lack of discipline": "auc_commitments_find_partners",
               },
               "protocols": {
                  "Not practical": [],
                  "Lack of discipline": [],
               },
            },

            "auc_commitments_reflect_on_understanding": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_commitments_reflect_on_understanding(user_id, app, db_session),

               "choices": {
                  "continue": "transfer_before_main_node",
               },
               "protocols": {
                  "continue": [],
               },
            },

            "auc_commitments_find_partners": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_commitments_find_partners(user_id, app, db_session),

               "choices": {
                  "continue": "transfer_before_main_node",
               },
               "protocols": {
                  "continue": [],
               },
            },

            "auc_commitments_write_plan": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_commitments_write_plan(user_id, app, db_session),

               "choices": {
                  "continue": "auc_commitments_compassion_energy_compassion",
               },
               "protocols": {
                  "continue": [],
               },
            },

            


            "auc_commitments_compassion_energy_compassion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_commitments_compassion_energy_compassion(user_id, app, db_session),

               "choices": {
                  #"continue": "auc_commitments_feel_accomplished",
                  "open_text": "calculate_compassion_energy",
                  
                  
               },
               "protocols": {
                  "open_text": [],
               },
            },





            "calculate_compassion_energy": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.calculate_compassion_energy(user_id),

               "choices": {
                  "continue": "auc_commitments_feel_accomplished",
               },
               "protocols": {
                  "continue": [],
               },
            },

            "auc_commitments_feel_accomplished": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_commitments_feel_accomplished(user_id, app, db_session),

               "choices": {
                  "yes": "auc_commitments_write_blog_post",
                  "no": "transfer_before_main_node",
               },
               "protocols": {
                 "yes": [],
                  "no": [],
               },
            },

            "auc_commitments_write_blog_post": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.auc_commitments_write_blog_post(user_id, app, db_session),

               "choices": {
                  "continue": "transfer_before_main_node",
               },
               "protocols": {
                 "continue": [],
               },
            },

            "ending_prompt": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_ending(user_id, app, db_session),

                "choices": {"any": "opening_prompt"},
                "protocols": {"any": []}
            },



            






            





        }
        self.QUESTION_KEYS = list(self.QUESTIONS.keys())


    # SHORTCUT: TEMPORARY
    
   
    # BEGINNING OF METHODS

    def initialise_prev_questions(self, user_id):
        self.recent_questions[user_id] = []

    def clear_persona(self, user_id):
        self.chosen_personas[user_id] = ""

    def clear_names(self, user_id):
        self.users_names[user_id] = ""

    def clear_datasets(self, user_id):
        self.datasets[user_id] = pd.DataFrame(columns=['sentences'])

    def initialise_remaining_choices(self, user_id):
        self.remaining_choices[user_id] = ["displaying_antisocial_behaviour", "internal_persecutor_saviour", "personal_crisis", "rigid_thought"]

    def save_name(self, user_id):
        try:
            user_response = self.user_choices[user_id]["choices_made"]["ask_name"]
        except:  # noqa
            user_response = ""
        self.users_names[user_id] = user_response
        return "greet_user"
        #return "opening_prompt"



    def get_suggestions(self, user_id, app): #from all the lists of protocols collected at each step of the dialogue it puts together some and returns these as suggestions
        suggestions = []
        for curr_suggestions in list(self.suggestions[user_id]):
            if len(curr_suggestions) > 2:
                i, j = random.choices(range(0,len(curr_suggestions)), k=2)
                if curr_suggestions[i] and curr_suggestions[j] in self.PROTOCOL_TITLES: #weeds out some gibberish that im not sure why it's there
                    suggestions.extend([curr_suggestions[i], curr_suggestions[j]])
            else:
                suggestions.extend(curr_suggestions)
            suggestions = set(suggestions)
            suggestions = list(suggestions)
        while len(suggestions) < 4: #augment the suggestions if less than 4, we add random ones avoiding repetitions
            p = random.choice([i for i in range(1,20) if i not in [6,11]]) #we dont want to suggest protocol 6 or 11 at random here
            if (any(self.PROTOCOL_TITLES[p] not in curr_suggestions for curr_suggestions in list(self.suggestions[user_id]))
                and self.PROTOCOL_TITLES[p] not in self.recent_protocols and self.PROTOCOL_TITLES[p] not in suggestions):
                        suggestions.append(self.PROTOCOL_TITLES[p])
                        self.suggestions[user_id].extend([self.PROTOCOL_TITLES[p]])
        return suggestions


    def clear_suggestions(self, user_id):
        self.suggestions[user_id] = []
        self.reordered_protocol_questions[user_id] = deque(maxlen=5)

    def clear_emotion_scores(self, user_id):
        self.guess_emotion_predictions[user_id] = ""

    def create_new_run(self, user_id, db_session, user_session):
        new_run = UserModelRun(session_id=user_session.id)
        db_session.add(new_run)
        db_session.commit()
        self.current_run_ids[user_id] = new_run.id
        return new_run

    def clear_choices(self, user_id):
        self.user_choices[user_id] = {}

    def update_suggestions(self, user_id, protocols, app):

        # Check if user_id already has suggestions
        try:
            self.suggestions[user_id]
        except KeyError:
            self.suggestions[user_id] = []

        if type(protocols) != list:
            self.suggestions[user_id].append(deque([protocols]))
        else:
            self.suggestions[user_id].append(deque(protocols))

    # Takes next item in queue, or moves on to suggestions
    # if all have been checked


    def greet_user(self, user_id):
        greet_user = ["Hello! Welcome to the CompassionBot! This chatbot is used to develop compassion (including foresighted compassion). Compassion is very important, and the absence of compassion can lead to severe consequences.",  \
                    "One example of lack of universal compassion is the Russian-Ukrainian war. In a series of events, leaders without compassion took actions that led to many people losing their lives. Another scenario is the ongoing conflict between Palestine and Israel. Lack of compassion has led to solutions being made on realpolitik, rather than the livelihood and peaceful co-living potential."]
        return greet_user

    def greet_user2(self, user_id):
        greet_user = ["Hence, the goal of the chatbot is to help develop compassion by forming an affectionate bonding with the childhood self. Enhancement of compassion can potentially play a role in helping us to overcome these world issues by transfer of compassion from childhood self to other people and contexts.", \
                    "More specifically, this chatbot aims to develop foresighted compassion.  Foresighted compassion goes beyond tender compassion, and aims to develop compassionate traits within you to develop sustainable, and actionable items for you. "]
                    
        return greet_user

    def get_opening_prompt(self, user_id):
        time.sleep(3)
        if self.users_names[user_id] == "":
            opening_prompt = [f"Hello, this is CompassionBot!" + ". ", "How are you feeling today?"]
        else:
            opening_prompt = ["Hello " + self.users_names[user_id] + ", this is CompassionBot!", "How are you feeling today?"]
        return opening_prompt


    def get_restart_prompt(self, user_id):
        time.sleep(3)
        if self.users_names[user_id] == "":
            restart_prompt = ["Please tell me again, how are you feeling today?"]
        else:
            restart_prompt = ["Please tell me again, " + self.users_names[user_id] + ", how are you feeling today?"]
        return restart_prompt

    def get_next_question(self, user_id):
        if self.remaining_choices[user_id] == []:
            return "project_emotion"
        else:
            selected_choice = np.random.choice(self.remaining_choices[user_id])
            self.remaining_choices[user_id].remove(selected_choice)
            return selected_choice

    def add_to_reordered_protocols(self, user_id, next_protocol):
        self.reordered_protocol_questions[user_id].append(next_protocol)

    def add_to_next_protocols(self, next_protocols):
        self.protocols_to_suggest.append(deque(next_protocols))

    def clear_suggested_protocols(self):
        self.protocols_to_suggest = []

    # NOTE: this is not currently used, but can be integrated to support
    # positive protocol suggestions (to avoid recent protocols).
    # You would need to add it in when a user's emotion is positive
    # and they have chosen a protocol.

    def add_to_recent_protocols(self, recent_protocol):
        if len(self.recent_protocols) == self.recent_protocols.maxlen:
            # Removes oldest protocol
            self.recent_protocols.popleft()
        self.recent_protocols.append(recent_protocol)


    def determine_next_prompt_opening(self, user_id, app, db_session):
        user_response = self.user_choices[user_id]["choices_made"]["opening_prompt"]
        emotion = get_emotion(user_response)
        #emotion = np.random.choice(["Happy", "Sad", "Angry", "Anxious"]) #random choice to be replaced with emotion classifier
        if emotion == 'fear':
            self.guess_emotion_predictions[user_id] = 'Anxious/Scared'
            self.user_emotions[user_id] = 'Anxious'
        elif emotion == 'sadness':
            self.guess_emotion_predictions[user_id] = 'Sad'
            self.user_emotions[user_id] = 'Sad'
        elif emotion == 'anger':
            self.guess_emotion_predictions[user_id] = 'Angry'
            self.user_emotions[user_id] = 'Angry'
        else:
            self.guess_emotion_predictions[user_id] = 'Happy/Content'
            self.user_emotions[user_id] = 'Happy'
        #self.guess_emotion_predictions[user_id] = emotion
        #self.user_emotions[user_id] = emotion
        return "guess_emotion"


    def get_best_sentence(self, column, prev_qs):
        #return random.choice(column.dropna().sample(n=15).to_list()) #using random choice instead of machine learning
        maxscore = 0
        chosen = ''
        # CHANGED_HERE from 5 to 2
        for row in column.dropna().sample(n=5): #was 25
             fitscore = get_sentence_score(row, prev_qs)
             if fitscore > maxscore:
                 maxscore = fitscore
                 chosen = row
        if chosen != '':
            return chosen
        else:
            return random.choice(column.dropna().sample(n=5).to_list()) #was 25


    def split_sentence(self, sentence):
        temp_list = re.split('(?<=[.?!]) +', sentence)
        if '' in temp_list:
            temp_list.remove('')
        temp_list = [i + " " if i[-1] in [".", "?", "!"] else i for i in temp_list]
        if len(temp_list) == 2:
            return temp_list[0], temp_list[1]
        elif len(temp_list) == 3:
            return temp_list[0], temp_list[1], temp_list[2]
        else:
            return sentence



    def get_model_prompt_guess_emotion(self, user_id, app, db_session):
        curr_question_code = "A02"
        emotion = f"From what you have said I believe you are feeling {self.guess_emotion_predictions[user_id].lower()}."
        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return [emotion, question]

    def get_model_prompt_check_emotion(self, user_id, app, db_session):
        curr_question_code = "A03"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column = data["All emotions - I am sorry. Please select from the emotions below the one that best reflects what you are feeling:"].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)

    def get_sad_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "Sad"
        self.user_emotions[user_id] = "Sad"
        return "after_classification_negative"
    def get_angry_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "Angry"
        self.user_emotions[user_id] = "Angry"
        return "after_classification_negative"
    def get_anxious_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "Anxious/Scared"
        self.user_emotions[user_id] = "Anxious"
        return "after_classification_negative"
    def get_happy_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "Happy/Content"
        self.user_emotions[user_id] = "Happy"
        return "after_classification_positive"

    def get_model_prompt_project_emotion(self, user_id, app, db_session):
        time.sleep(7)
        if self.chosen_personas[user_id] == "Robert":
            prompt = "Ok, thank you. Now, one last important thing: since you've told me you're feeling " + self.user_emotions[user_id].lower() + ", I would like you to try to project this emotion onto your childhood self. You can press 'continue' when you are ready and I'll suggest some protocols I think may be appropriate for you."
        elif self.chosen_personas[user_id] == "Gabrielle":
            prompt = "Thank you, I will recommend some protocols for you in a moment. Before I do that, could you please try to project your " + self.user_emotions[user_id].lower() + " feeling onto your childhood self? Take your time to try this, and press 'continue' when you feel ready."
        elif self.chosen_personas[user_id] == "Arman":
            prompt = "Ok, thank you for letting me know that. Before I give you some protocol suggestions, please take some time to project your current " + self.user_emotions[user_id].lower() + " feeling onto your childhood self. Press 'continue' when you feel able to do it."
        elif self.chosen_personas[user_id] == "Arman":
            prompt = "Ok, thank you, I'm going to draw up a list of protocols which I think would be suitable for you today. In the meantime, going back to this " + self.user_emotions[user_id].lower() + " feeling of yours, would you like to try to project it onto your childhood self? You can try now and press 'continue' when you feel ready."
        else:
            prompt = "Thank you. While I have a think about which protocols would be best for you, please take your time now and try to project your current " + self.user_emotions[user_id].lower() + " emotion onto your childhood self. When you are able to do this, please press 'continue' to receive your suggestions."
        return self.split_sentence(prompt)

    def get_model_overcome_negative_emotions(self, user_id, app, db_session):
        prompt = "I am sorry to hear that you are not feeling the best. Let us do Exercise 13 to overcome current negative emotions." 
        return prompt


    def get_model_prompt_saviour(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + " - Do you believe that you should be the saviour of someone else?"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_victim(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + " - Do you see yourself as the victim, blaming someone else for how negative you feel?"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_controlling(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + " - Do you feel that you are trying to control someone?"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_accusing(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + " - Are you always blaming and accusing yourself for when something goes wrong?"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_specific_event(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + " - Was this caused by a specific event/s?"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_event_is_recent(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + " - Was this caused by a recent or distant event (or events)?"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_revisit_recent(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + " - Have you recently attempted protocol 11 and found this reignited unmanageable emotions as a result of old events?"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_revisit_distant(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + " - Have you recently attempted protocol 6 and found this reignited unmanageable emotions as a result of old events?"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_more_questions(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + " - Thank you. Now I will ask some questions to understand your situation."
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_antisocial(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + " - Have you strongly felt or expressed any of the following emotions towards someone:"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return [self.split_sentence(question), "Envy, jealousy, greed, hatred, mistrust, malevolence, or revengefulness?"]

    def get_model_prompt_rigid_thought(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + " - In previous conversations, have you considered other viewpoints presented?"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_personal_crisis(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + " - Are you undergoing a personal crisis (experiencing difficulties with loved ones e.g. falling out with friends)?"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_happy(self, user_id, app, db_session):
        """
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column = data["Happy - That's Good! Let me recommend a protocol you can attempt."].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        #return self.split_sentence(question)
        """
        return ["I am really glad to hear that that you're happy. Let us go to the main node."]

    def get_model_prompt_suggestions(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column = data["All emotions - Here are my recommendations, please select the protocol that you would like to attempt"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)


    # COMPASSION METHOD
    def trying_protocol_9(self, user_id, app, db_session):
        curr_question_code = "A04"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
    
    

    
    def get_model_prompt_found_compassion_to_child(self, user_id, app, db_session):
        curr_question_code = "A05"

        additional_prompt = "Now, project your compassion towards your childhood self."
        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return [additional_prompt, question]
        #return ["Now take a deep breath. Close your eyes, and feel. Do you feel compassionate to your childhood self?"]

    def user_found_no_compassion_to_child_after_3(self, user_id, app, db_session):
        return ["It's okay! This is exactly why we need to practise more on SAT! Take some time to go through SAT later at MAIN NODE."]
    
    def check_tender_vs_foresighted_compassion(self, user_id, app, db_session):
        curr_question_code = "A06"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        #return ["Do you feel like you want to protect the child, or do you feel like you need to help the child to grow up?"]

    def shown_tender_compassion(self, user_id, app, db_session):
        curr_question_code = "A07"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        curr_question_code = "A09"

        question2 = self.get_best_sentence_from_question_code(user_id, curr_question_code)
        tender_compassion = "The more direct form of compassion - tender compassion has been researched extensively by Professor Paul Gilbert. You can look it up to learn more about the amazing work done by Professor Gilbert."
        
        return [question, tender_compassion, question2]
        return ["Congratulations! You have shown tender compassion. Tender compassion is the first phase of compassion, where you feel sympathy towards the victim. \
                You feel for the victim, hoping to do something that can alleviate the suffering of the victim immediately. \
                Now, it is time to bring the compassion to the next level. Please take SAT module later to help you develop foresighted compassion! Foresighted compassion \
                aims to help you develop long-term solutions that can help victim to face the suffering , and grow from there. Let's go further!"]
    
    
    def shown_foresighted_compassion(self, user_id, app, db_session):
        curr_question_code = "A08"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        curr_question_code = "A10"

        question2 = self.get_best_sentence_from_question_code(user_id, curr_question_code)


        return [question, question2]
        return ["Congratulations! You have shown foresighted compassion to your childhood self! Contrary to tender compassion which is an immediate form of sympathy towards victim \
            you have shown foresighted compassion which helps you to think how to help victim long-term! Now, it is time for you to use your compassion \
                to do meaningful things in the real world! At the MAIN NODE, please go to Awareness, Understanding, Commitments (AUC) to start making impact in the real world!"]
    
    def shown_compassion_ask_feel_better(self, user_id, app, db_session):
        curr_question_code = "A08"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question

    # SHORTCUT: MAIN NODE METHOD 
    def main_node(self, user_id, app, db_session):
        main_node_msg= "Welcome to the MAIN NODE! What do you feel like doing today?"
        return ["Welcome to the MAIN NODE! What do you feel like doing today?"]

    def main_node_faq(self, user_id, app, db_session):
        main_node_instruction = "If you are new to the chatbot, you should start by going to SAT to practice SAT."
        main_node_instruction2 = "After going through a round of practises, go to ESA to practice tender compassion."
        main_node_instruction3 = "Finaly, go to AUC to expose yourself to world issues and practice your newly founded compassion! You will then be asked to go to COI to discuss some of the ideas."
        return [main_node_instruction, main_node_instruction2, main_node_instruction3]

    def congrats_before_main_node(self, user_id, app, db_session):
        return ["Congratulations! Now let us move to MAIN NODE."]

    def transfer_before_main_node(self, user_id, app, db_session):
        return ["You have done great so far! Now let us move to MAIN NODE."]

    def come_back_tomorrow_before_main_node(self, user_id, app, db_session):
        return ["Maybe today is not working so well.", "Take some rest, and come back to this tomorrow.", "Let us go back to MAIN NODE."]

    def coi_start(self, user_id, app, db_session):
        coi_welcome = "Welcome to Clash of Ideas (COI)!"
        coi_msg = "COI is an alternative blog specifically created to allow the anonymous exchange of ideas. "
        coi_msg2 = "The main idea is to learn from other people, and not judge others' ideas and actions no matter how small or big, and hence, no comment is allowed."
        return [coi_welcome, coi_msg, coi_msg2]


    def coi_start2a(self, user_id, app_db, session):
        answer_template_invite = "You can use the following template when answering the question that you have in mind."
        answer_template = """Problem -> Existing solutions -> Proposed solutions -> Feasibility \
                         -> Resources required -> Commitments from yourself -> Evaluation metrics on success -> Feeling about compassion"""
        answer_link = f"Please go to {COI_IP_ADDRESS} for Clash of Ideas (COI)."
        return [answer_template_invite, answer_template, answer_link]
        
    def coi_start2b(self, user_id, app, db_session):
        answer_link = f"Please go to {COI_IP_ADDRESS} for Clash of Ideas (COI)."
        return [answer_link]



    # SHORTCUT: ESA
    def esa_last_week(self, user_id, app, db_session):
        curr_question_code = "G01"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Do you follow through last week actions?"]

    def esa_set_count(self, user_id, db_session, action):
        user_commitments = UserCommitment.query.filter_by(user_id=user_id).first()
        if user_commitments == None:
            user_commitments = UserCommitment(user_id=user_id)
            db_session.add(user_commitments)
            db_session.commit()

        if action == "yes":
            user_commitments.esa_action_count += 1
        user_commitments.esa_total_count += 1
        db_session.commit()
        return "esa_show_statistics"

    def esa_show_statistics(self, user_id):
        user_commitments = UserCommitment.query.filter_by(user_id=user_id).first()

        completed_action = user_commitments.esa_action_count
        total = user_commitments.esa_total_count
        completion_perc = completed_action * 100/total
        completion_perc = "{:.2f}".format(completion_perc)
        statistics_format = ["Allow me to share your completion statistics!",
                            f"Completed: {completed_action}", \
                              f"Total: {total}", \
                              f"Percentage of completion: {completion_perc}%"]
        return statistics_format

    def esa_simple_scenario(self, user_id):
        congrats_user = "Let us start with this week's exercise."
        simple_scenarios = ["Before you argue with the next person today, slow down, calm and listen to the person first.\
                            Don’t try to give your reasons immediately, let them know that you will give them a reply the next day. \
                            Take your time, think about it before you sleep, and give a proper response the next day", \
                            "Have you judged someone when they said or write something recently? This time, try not to judge, try to accept the person as who he/she is.",\
                            "If you are busy, before you brush someone off today, try to look into their eyes, listen attentively, and make them feel your presence. "]
        chosen_scenario =  random.choice(simple_scenarios)
        msg = congrats_user + chosen_scenario
        return self.split_sentence(msg)
    
    # SHORTCUT: SAT
    def sat_compassion_start(self, user_id, app, db_session):
        curr_question_code = "B01"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Before we start, notes that the main purpose of using this chatbot is to develop compassion. "]
        
    def sat_compassion_difference(self, user_id, app, db_session):
        curr_question_code = "B02"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Understand that there are 2 types of compassion. Tender compassion is a more commonly seen compassion, where the people generates immediate sympathetic feeling towards some unfortunate events. Another form of compassion, which goes beyond initial sympathetic feeling, and wanted to do more sustainable and systematically to alleviate the situation, would be known as foresighted compassion. Do you need an example?"]

    def sat_compassion_difference_example(self, user_id, app, db_session):
        tender_compassion = "Tender compassion is the most direct form of compassion. People with tender compassion will often provide immediate assistance towards someone who needs it immediately, such as giving donations, food, etc."
        tender_compassion2 = "Tender compassion has been researched extensively by Professor Paul Gilbert. You can look it up to learn more about the amazing work done by Professor Gilbert."
        foresighted_compassion = "On the other hand, foresighted compassion is a more holistic, and indirect form of compassion. People with foresighted compassion will spend time thinking holistically about a problem, thinking about the potential root cause, and aims to solve it systematically."
        return [tender_compassion, tender_compassion2, foresighted_compassion]


    def project_emotion_to_child(self, user_id, app, db_session):
        return ["Let us continue. Now, close your eyes, project the emotion to the child, and feel the emotion."]
    
    def sat_feel_compassion_to_child_question(self, user_id, app, db_session):
        curr_question_code = "A05"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Do you feel compassionate towards your childhood self?"]

    # SHORTCUT: SAT PART 1
    def sat_part1_begin(self, user_id, app, db_session):
        curr_question_code = "B03"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["This is fine! This is exactly the reason this chatbot is created! We aim to develop compassion for you! Let us begin!"]
    
    def trying_protocol_1_and_2(self, user_id, app, db_session):
        curr_question_code = "B04"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["To feel compassionate towards the child, we have to first become intimate with our child. Let us practise exercise 1 and 2."]

    def sat_form_connection_with_child_level1(self, user_id, app, db_session):
        curr_question_code = "B05"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Do you form some connection with your childhood self?"]
    
    def sat_take_child_relationship_to_next_level(self, user_id, app, db_session):
        curr_question_code = "B06"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Amazing! You have accomplished the first goal! However, we are still not exactly there. Are you ready to take the relationship with the child self to the next level? "]
    
    def sat_go_esa_or_end(self, user_id, app, db_session):
        curr_question_code = "B07"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question

    def trying_protocol_3_and_4_and_5(self, user_id, app, db_session):
        curr_question_code = "B08"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Now that you feel your connection with the child, time to show your childhood self some love. Exercise 3, 4 and 5."]
    
    def sat_form_connection_with_child_level2(self, user_id, app, db_session):
        curr_question_code = "B09"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Do you feel love and feel like you want to protect your child now?"]

    def sat_completed_tender_compassion_training(self, user_id, app, db_session):
        curr_question_code = "B10"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Congratulations! You have shown tender compassion! This is a great first stage of compassion, where you have shown sympathy towards the child! Now, let us go on and see what to do with this compassion."]

    # SHORTCUT: SAT PART 2

    def sat_ask_type_of_compassion(self, user_id, app, db_session):
        curr_question_code = "C01"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["What kind of compassion do you feel towards your childhood self? Is that tender compassion which is an immediate sympathy towards the child, or is that foresighted compassion, where you help to make the child grow for the long term?"]
    
    def sat_shown_tender_compassion(self, user_id, app, db_session):
        curr_question_code = "C03"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["You have shown tender compassion towards the childhood self. Now, try to shift that compassion, and project that compassion feeling towards a vulnerable community (If you need help thinking about a vulnerable community, go to AUC)."]

    def sat_ask_if_help_vulnerable_communities(self, user_id, app, db_session):
        curr_question_code = "C04"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Do you feel like you should help them?"]
    
    def sat_ask_why_not_help_vulnerable_communities(self, user_id, app, db_session):
        curr_question_code = "C05"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Why is that? Do you feel like they do not deserve some help, or you can't feel the compassion towards them?"]
    
    def sat_answer_cant_feel_compassion(self, user_id, app, db_session):
        curr_question_code = "C06"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Let us take a step back, and try to form an intimate relationship with our childhood self."]
    
    def trying_protocol_15(self, user_id, app, db_session):
        curr_question_code = "C07"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Let us take a step back. With the tender compassion that you have developed towards your child, try to do exercise 15 (change perspective)."]
    
    def sat_imagine_self_vulnerable(self, user_id, app, db_session):
        curr_question_code = "C08"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Now close your eyes, imagine that you're part of that vulnerable community. Do you feel you deserve some help?"]

    def sat_ask_if_help_vulnerable_communities2(self, user_id, app, db_session):
        curr_question_code = "C10"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Now back to your own self. Do you feel like you should help the vulnerable community?"]

    def sat_vulnerable_communities_think_deeper(self, user_id, app, db_session):
        curr_question_code = "C09"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["There must be something deeper that lead the position of them as the vulnerable communities today. Think deeper. Now, do you want to help them?"]
    
    
    # SHORTCUT: SAT PART 3

    
    def sat_ask_user_if_want_protocol_8(self, user_id, app, db_session):
        curr_question_code = "D01"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Great! If you believe that you have shown the trait of foresighted compassion, is it okay if still do some protocols to see how we do?"]

    def trying_protocol_8(self, user_id, app, db_session):
        curr_question_code = "D02"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Let us to exercise 8. When you are done, click on continue."]

    # SHORTCUT: SAT PART 4
    def sat_how_to_help_vulnerable_community(self, user_id, app, db_session):
        curr_question_code = "E01"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["How do you feel like you should help them?"]
    
    def sat_shown_foresighted_compassion_and_ready(self, user_id, app, db_session):
        curr_question_code = "E02"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Congratulations! You have shown foresighted compassion, where you expect the child to deal with situation maturely. You have developed the capability to develop long-term sustainable solutions. Let us move to MAIN NODE to make meaningful actions."]
    
    def trying_protocol_17_and_18(self, user_id, app, db_session):
        curr_question_code = "E03"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Now think about this. Sometimes child make unreasonable requests. When they don't get it, they will be upset. It is the responsibility of our adult self to teach them to bear with it, deal with the situation with maturity. Let us practise exercise 17 and 18."]
    def sat_immediate_action_or_maturity(self, user_id, app, db_session):
        curr_question_code = "E04"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Do you feel that the best sort of action is perhaps not something you immediately think of, or something that the child want, rather, it could be a more difficult action that requires discipline or maturity?"]
    
    def sat_completed_foresighted_compassion_training(self, user_id, app, db_session):
        curr_question_code = "E05"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Congratulations! You have shown foresighted compassion. Now it is time to practise it! Head to AUC so that you can do something meaningful! Head to MAIN NODE."]


    # SHORTCUT: SAT PART 5
    def sat_might_be_anti_social(self, user_id, app, db_session):
        curr_question_code = "F01"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["There might be some form of anti-social behaviour exhibited. Let us dives a little bit deeper."]

    def sat_personal_resentment(self, user_id, app, db_session):
        curr_question_code = "F02"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Do you have some personal resentments towards someone you know?"]

    def trying_protocol_17_and_18_anti_social(self, user_id, app, db_session):
        curr_question_code = "F08"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["There is a different degree of anti-social behavior. Let us take some time to act it out. Do exercise 17 and 18."]

    def sat_ask_care_social(self, user_id, app, db_session):
        curr_question_code = "F09"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Do you feel you care a little bit more about social environments?"]

    def sat_care_about_other_people(self, user_id, app, db_session):
        curr_question_code = "F03"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Do you care about things that happening around you? For example, friends, families?"]

    def sat_local_community(self, user_id, app, db_session):
        curr_question_code = "F04"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Do you know what is happening in your local community? Have you joined any local activity?"]

    def sat_global_community(self, user_id, app, db_session):
        curr_question_code = "F05"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Do you care about events happening at the other side of the world?"]

    def sat_global_community_compassion(self, user_id, app, db_session):
        curr_question_code = "F06"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Great! Do you feel compassionate towards events happening at the other side of the world"]

    def sat_no_anti_social(self, user_id, app, db_session):
        curr_question_code = "F07"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
        return ["Amazing! Now let us go back to the question about vulnerable community. "]
    

    # SHORTCUT: AUC: Awareness

    def auc_choose_a_u_c(self, user_id, app, db_session):
        question = "Welcome to Awareness, Understanding, Commitments (AUC)."
        question2 = "The goal of AUC is to improve exposure to relevant world issues through Awareness. With compassion in mind, we will try to understand the issues, obstacles and existing solutions by transferring the projection of compassion from childhood self to the issue. We then try to make a commitment by making small actions to help solve the issue."
        question3 = "If you are ready, please select the stage you are in!"
        return [question, question2, question3]

    def auc_awareness_begin(self, user_id, app, db_session):
        curr_question_code = "H01"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question

    def auc_awareness_get_another_news(self, user_id, app, db_session):
        question = "Perhaps we have started too far! Let us try with another category."
        question2 = "You can also choose back the same category, we will get different news!"

        return [question, question2]
        
    

    def auc_awareness_set_news(self, user_id, app, db_session, action):
        if user_id not in self.nodes_direction.keys():
            self.nodes_direction[user_id] = {"news_type": action}

        else:
            self.nodes_direction[user_id] = {"news_type": action}
        return "auc_awareness_get_news"

    def auc_awareness_get_news(self, user_id, app, db_session):
        news_type = self.nodes_direction[user_id]["news_type"]
        df = self.news_by_category[news_type]
        random_news = df.sample().values
        msg = "Let us start by reading this news. With compassion in mind, click on the link, and read it."
        try:
            title = random_news[0][1]
            content = random_news[0][2]
            url = random_news[0][3]
            return [msg, title, content, url]
        except:
            url = random_news[0][1]
            return [msg, url]

        return [title, content, url]



    def auc_awareness_feel_reading_news(self, user_id, app, db_session):
        curr_question_code = "H02"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question

    def auc_awareness_no_feeling(self, user_id, app, db_session):
        curr_question_code = "H03"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question

    def auc_awareness_go_practice_sat(self, user_id, app, db_session):
        curr_question_code = "H04"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question

    def auc_awareness_feel_compassion(self, user_id, app, db_session):
        curr_question_code = "H05"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question

    def auc_awareness_quick_test(self, user_id, app, db_session):
        curr_question_code = "H06"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question

    def auc_awareness_explain_compassion_difference(self, user_id, app, db_session):
        curr_question_code = "H07"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question

    def auc_awareness_congrats(self, user_id, app, db_session):
        curr_question_code = "H08"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question

    # SHORTCUT: AUC: Understanding
    def auc_understanding_begin(self, user_id, app, db_session):
        question =  "Welcome to part 2 of AUC, researching deeper about existing solutions (Understanding)!"

        return question

        
    def auc_understanding_research_existing(self, user_id, app, db_session):
        curr_question_code = "I01"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question

    def auc_understanding_research_set_count(self, user_id, db_session, action):
        if user_id not in self.nodes_direction.keys():
            self.nodes_direction[user_id] = {"I01": "na"}


        user_commitments = UserCommitment.query.filter_by(user_id=user_id).first()
        if user_commitments == None:
            user_commitments = UserCommitment(user_id=user_id)
            db_session.add(user_commitments)
            db_session.commit()

        if action == "yes":
            user_commitments.auc_understanding_action_count += 1
            self.nodes_direction[user_id]["I01"] = "yes"
        else:
            self.nodes_direction[user_id]["I01"] = "no"

        user_commitments.auc_understanding_total_count += 1
        db_session.commit()
        return "auc_understanding_research_statistics"

    def save_calculate_compassion_energy_free_text(self, user_id, free_text):
        pass
    def calculate_compassion_energy(self, user_id):
        #(w1 * (consistency_score/100)  + w2 * (compassion_score + 1/3))/  (w1 + w2)
        user_commitments = UserCommitment.query.filter_by(user_id=user_id).first()

        try:
            completed_action = user_commitments.auc_understanding_action_count
            total = user_commitments.auc_understanding_total_count
            auc_understanding_consistency = completed_action/total

            completed_action = user_commitments.auc_understanding_action_count
            total = user_commitments.auc_understanding_total_count
            auc_commitments_consistency = completed_action/total

            completed_action = user_commitments.esa_action_count
            total = user_commitments.esa_total_count
            esa_consistency = completed_action/total

            print(self.user_choices[user_id]["choices_made"])
            user_response = self.user_choices[user_id]["choices_made"]["auc_commitments_compassion_energy_compassion"]
            empathy_level = empathy_score(user_response)


            w1 = self.compassion_energy_weightage['esa_consistency']
            w2 = self.compassion_energy_weightage['auc_understanding_consistency']
            w3 = self.compassion_energy_weightage['auc_commitments_consistency']
            w4 = self.compassion_energy_weightage['empathetic_score']
            w_total = w1 + w2 + w3 + w4


            compassion_energy_score = ((w1 * esa_consistency) + \
                                (w2 * auc_understanding_consistency) + \
                                (w3 * auc_commitments_consistency) + \
                                (w4 * (empathy_level + 1)/3)) * 100/ w_total

            compassion_energy_score = "{:.2f}".format(compassion_energy_score)
            return [f'We have estimated that you have shown approximately {compassion_energy_score}% of compassion energy!']

        except:
            return ["You have yet to complete ESA and AUC. Complete them at least once, and we will be able to calculate your compassion energy!"]
        






        total = user_commitments.auc_understanding_total_count
    def auc_understanding_research_statistics(self, user_id):
        user_commitments = UserCommitment.query.filter_by(user_id=user_id).first()

        completed_action = user_commitments.auc_understanding_action_count
        total = user_commitments.auc_understanding_total_count
        completion_perc = completed_action * 100/total
        completion_perc = "{:.2f}".format(completion_perc)
        statistics_format = ["Allow me to share your completion statistics!",
                            f"Completed: {completed_action}", \
                              f"Total: {total}", \
                              f"Percentage of completion: {completion_perc}%"]
        return statistics_format

    def auc_understanding_research_decide_direction(self, user_id, db_session):
        if self.nodes_direction[user_id]["I01"] == "yes":
            return "auc_understanding_research_reasonable_solutions"
        else:
            return "auc_understanding_set_deadline"



    def auc_understanding_set_deadline(self, user_id, app, db_session):
        curr_question_code = "I02"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question

    def auc_understanding_research_reasonable_solutions(self, user_id, app, db_session):
        curr_question_code = "I03"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question

    def auc_understanding_go_coi(self, user_id, app, db_session):
        question = "Time to spark some ideas! Let us go to Clash of Ideas (COI) to discuss ideas with others."

        return question

    def auc_understanding_write_plan(self, user_id, app, db_session):
        curr_question_code = "I04"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question

    # SHORTCUT: AUC: Commitments

    def auc_commitments_begin(self, user_id, app, db_session):
        question =  "Welcome to part 3 of AUC, making meaningful actions to practise compassion (Commitments)!"
        return question

    def auc_commitments_follow_through_solutions(self, user_id, app, db_session):
        curr_question_code = "J01"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question

    def auc_commitments_research_set_count(self, user_id, db_session, action):
        if user_id not in self.nodes_direction.keys():
            self.nodes_direction[user_id] = {"J01": "na"}

        user_commitments = UserCommitment.query.filter_by(user_id=user_id).first()
        if user_commitments == None:
            user_commitments = UserCommitment(user_id=user_id)
            db_session.add(user_commitments)
            db_session.commit()

        if action == "yes":
            user_commitments.auc_commitments_action_count += 1
            self.nodes_direction[user_id]["J01"] = "yes"
        else:
            self.nodes_direction[user_id]["J01"] = "no"
        
        user_commitments.auc_commitments_total_count += 1
        db_session.commit()
        return "auc_commitments_research_statistics"

    def auc_commitments_research_statistics(self, user_id):
        user_commitments = UserCommitment.query.filter_by(user_id=user_id).first()

        completed_action = user_commitments.auc_commitments_action_count
        total = user_commitments.auc_commitments_total_count
        completion_perc = completed_action * 100/total
        completion_perc = "{:.2f}".format(completion_perc)
        statistics_format = ["Allow me to share your completion statistics!",
                            f"Completed: {completed_action}", \
                              f"Total: {total}", \
                              f"Percentage of completion: {completion_perc}%"]
        return statistics_format

    def auc_commitments_research_decide_direction(self, user_id, db_session):
        if self.nodes_direction[user_id]["J01"] == "yes":
            return "auc_commitments_write_plan"
        else:
            return "auc_commitments_why_not_follow_through_solutions"

    def auc_commitments_why_not_follow_through_solutions(self, user_id, app, db_session):
        question =  "May I know why you did not follow through with the solution?"

        return question

    def auc_commitments_reflect_on_understanding(self, user_id, app, db_session):
        curr_question_code = "J02"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question
    
    def auc_commitments_find_partners(self, user_id, app, db_session):
        curr_question_code = "J03"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question

    def auc_commitments_write_plan(self, user_id, app, db_session):
        question = "Please write down the following:"
        action = "Action -> Impact -> What can be improved."
        return [question, action]

    def auc_commitments_compassion_energy_compassion(self, user_id, app, db_session):
        curr_question_code = "J04"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question

    def auc_commitments_feel_accomplished(self, user_id, app, db_session):
        question = "Do you feel accomplished?"

        return question

    def auc_commitments_write_blog_post(self, user_id, app, db_session):
        curr_question_code = "J05"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question

    def get_model_prompt_ending(self, user_id, app, db_session):
        curr_question_code = "K01"

        question = self.get_best_sentence_from_question_code(user_id, curr_question_code)

        return question

    def update_conversation(self, user_id, new_dialogue, db_session, app):
        try:
            session_id = self.user_choices[user_id]["current_session_id"]
            curr_session = UserModelSession.query.filter_by(id=session_id).first()
            if curr_session.conversation is None:
                curr_session.conversation = "" + new_dialogue
            else:
                curr_session.conversation = curr_session.conversation + new_dialogue
            curr_session.last_updated = datetime.datetime.utcnow()
            db_session.commit()
        except KeyError:
            curr_session = UserModelSession(
                user_id=user_id,
                conversation=new_dialogue,
                last_updated=datetime.datetime.utcnow(),
            )

            db_session.add(curr_session)
            db_session.commit()
            self.user_choices[user_id]["current_session_id"] = curr_session.id


    def save_current_choice(
        self, user_id, input_type, user_choice, user_session, db_session, app
    ):
        # Set up dictionary if not set up already
        # with Session() as session:

        try:
            self.user_choices[user_id]
        except KeyError:
            self.user_choices[user_id] = {}

        # Define default choice if not already set
        try:
            current_choice = self.user_choices[user_id]["choices_made"][
                "current_choice"
            ]
        except KeyError:
            current_choice = self.QUESTION_KEYS[0]

        try:
            self.user_choices[user_id]["choices_made"]
        except KeyError:
            self.user_choices[user_id]["choices_made"] = {}

        if current_choice == "ask_name":
            self.clear_suggestions(user_id)
            self.user_choices[user_id]["choices_made"] = {}
            self.create_new_run(user_id, db_session, user_session)

        # Save current choice
        self.user_choices[user_id]["choices_made"]["current_choice"] = current_choice
        self.user_choices[user_id]["choices_made"][current_choice] = user_choice

        curr_prompt = self.QUESTIONS[current_choice]["model_prompt"]
        # prompt_to_use = curr_prompt
        if callable(curr_prompt):
            curr_prompt = curr_prompt(user_id, db_session, user_session, app)

        #removed stuff here

        else:
            self.update_conversation(
                user_id,
                "Model:{} \nUser:{} \n".format(curr_prompt, user_choice),
                db_session,
                app,
            )

        # ADD IN WHEN SHOWING PROTOCOLS
        if (
            current_choice == "after_classification_negative"
        ):
            try:
                current_protocol = self.TITLE_TO_PROTOCOL[user_choice]
            except KeyError:
                current_protocol = int(user_choice)

            protocol_chosen = Protocol(
                protocol_chosen=current_protocol,
                user_id=user_id,
                session_id=user_session.id,
                run_id=self.current_run_ids[user_id],
            )
            db_session.add(protocol_chosen)
            db_session.commit()
            self.current_protocol_ids[user_id] = [current_protocol, protocol_chosen.id]

            for i in range(len(self.suggestions[user_id])):
                curr_protocols = self.suggestions[user_id][i]
                if curr_protocols[0] == self.PROTOCOL_TITLES[current_protocol]:
                    curr_protocols.popleft()
                    if len(curr_protocols) == 0:
                        self.suggestions[user_id].pop(i)
                    break


        # Case: update suggestions for next attempt by removing relevant one
        # ADD IN WHEN SHOWING PROTOCOLS
        elif (
            current_choice == "suggestions"
        ):

            # PRE: user_choice is a string representing a number from 1-20,
            # or the title for the corresponding protocol

            try:
                current_protocol = self.TITLE_TO_PROTOCOL[user_choice]
            except KeyError:
                current_protocol = int(user_choice)

            protocol_chosen = Protocol(
                protocol_chosen=current_protocol,
                user_id=user_id,
                session_id=user_session.id,
                run_id=self.current_run_ids[user_id],
            )
            db_session.add(protocol_chosen)
            db_session.commit()
            self.current_protocol_ids[user_id] = [current_protocol, protocol_chosen.id]

            for i in range(len(self.suggestions[user_id])):
                curr_protocols = self.suggestions[user_id][i]
                if curr_protocols[0] == self.PROTOCOL_TITLES[current_protocol]:
                    curr_protocols.popleft()
                    if len(curr_protocols) == 0:
                        self.suggestions[user_id].pop(i)
                    break

        # PRE: User choice is string in ["Better", "Worse"]
        elif current_choice == "user_found_useful":
            current_protocol = Protocol.query.filter_by(
                id=self.current_protocol_ids[user_id][1]
            ).first()
            current_protocol.protocol_was_useful = user_choice
            db_session.commit()

        if current_choice == "guess_emotion":
            option_chosen = user_choice + " ({})".format(
                self.guess_emotion_predictions[user_id]
            )
        else:
            option_chosen = user_choice
        choice_made = Choice(
            choice_desc=current_choice,
            option_chosen=option_chosen,
            user_id=user_id,
            session_id=user_session.id,
            run_id=self.current_run_ids[user_id],
        )
        db_session.add(choice_made)
        db_session.commit()

        return choice_made

    def determine_next_choice(
        self, user_id, input_type, user_choice, db_session, user_session, app
    ):
        # Find relevant user info by using user_id as key in dict.
        #
        # Then using the current choice and user input, we determine what the next
        # choice is and return this as the output.

        # Some edge cases to consider based on the different types of each field:
        # May need to return list of model responses. For next protocol, may need
        # to call function if callable.

        # If we cannot find the specific choice (or if None etc.) can set user_choice
        # to "any".

        # PRE: Will be defined by save_current_choice if it did not already exist.
        # (so cannot be None)

        current_choice = self.user_choices[user_id]["choices_made"]["current_choice"]
        current_choice_for_question = self.QUESTIONS[current_choice]["choices"]
        current_protocols = self.QUESTIONS[current_choice]["protocols"]

        print("====== Determine next choice (rule_based_model.py) ======")
        print(f"current_choice: {current_choice}") # choose_persona/ ask_name
        print(f"user_choice: {user_choice}") # kai /wj
        print(f"current_choice_for_question: {current_choice_for_question}") #{kai: <function>, robert, ..} /{open_text: <function>}
        print(f"current_protocols: {current_protocols}") #{kai: [], robert: [], ..} /{open_text: []}
        print(f"input_type: {input_type}") # protocol / open_text
        print("================ Ended ==================")

        if input_type != "open_text":
            if (
                current_choice != "suggestions"
                and current_choice != "event_is_recent"
                and current_choice != "more_questions"
                #and current_choice != "after_classification_positive"
                and current_choice != "user_found_useful"
                and current_choice != "check_emotion"
                and current_choice != "new_protocol_better"
                and current_choice != "new_protocol_worse"
                and current_choice != "new_protocol_same"
                and current_choice != "choose_persona"
                and current_choice != "project_emotion"
                #and current_choice != "after_classification_negative"
                and current_choice != "greet_user"
                and current_choice != "main_node"
                and current_choice != "sat_how_to_help_vulnerable_community"
                and current_choice != "esa_simple_scenario"
                and current_choice != "auc_choose_a_u_c"
                and current_choice != "sat_ask_why_not_help_vulnerable_communities"
                and current_choice not in self.no_lowercase_node
            ):
                
                user_choice = user_choice.lower()
            if (
                current_choice == "suggestions"
            ):
                try:
                    current_protocol = self.TITLE_TO_PROTOCOL[user_choice]
                except KeyError:
                    current_protocol = int(user_choice)
                protocol_choice = self.PROTOCOL_TITLES[current_protocol]
                next_choice = current_choice_for_question[protocol_choice]
                protocols_chosen = current_protocols[protocol_choice]

            #elif current_choice == "greet_user":
            #    next_choice 

            elif current_choice == "check_emotion":
                if user_choice == "Sad":
                    next_choice = current_choice_for_question["Sad"]
                    protocols_chosen = current_protocols["Sad"]
                elif user_choice == "Angry":
                    next_choice = current_choice_for_question["Angry"]
                    protocols_chosen = current_protocols["Angry"]
                elif user_choice == "Anxious/Scared":
                    next_choice = current_choice_for_question["Anxious/Scared"]
                    protocols_chosen = current_protocols["Anxious/Scared"]
                else:
                    next_choice = current_choice_for_question["Happy/Content"]
                    protocols_chosen = current_protocols["Happy/Content"]
            else:
                next_choice = current_choice_for_question[user_choice]
                protocols_chosen = current_protocols[user_choice]

        else:
            next_choice = current_choice_for_question["open_text"]
            protocols_chosen = current_protocols["open_text"]

        if callable(next_choice):
            next_choice = next_choice(user_id, db_session, user_session, app)

        if current_choice == "guess_emotion" and user_choice.lower() == "yes":
            if self.guess_emotion_predictions[user_id] == "Sad":
                next_choice = next_choice["Sad"]
            elif self.guess_emotion_predictions[user_id] == "Angry":
                next_choice = next_choice["Angry"]
            elif self.guess_emotion_predictions[user_id] == "Anxious/Scared":
                next_choice = next_choice["Anxious/Scared"]
            else:
                next_choice = next_choice["Happy/Content"]

        if callable(protocols_chosen):
            protocols_chosen = protocols_chosen(user_id, db_session, user_session, app)
        next_prompt = self.QUESTIONS[next_choice]["model_prompt"]
        if callable(next_prompt):
            next_prompt = next_prompt(user_id, db_session, user_session, app)
        if (
            len(protocols_chosen) > 0
            and current_choice != "suggestions"
        ):
            self.update_suggestions(user_id, protocols_chosen, app)

        # Case: new suggestions being created after first protocol attempted
        if next_choice == "opening_prompt":
            self.clear_suggestions(user_id)
            self.clear_emotion_scores(user_id)
            self.create_new_run(user_id, db_session, user_session)

        if next_choice == "suggestions":
            next_choices = self.get_suggestions(user_id, app)

        else:
            next_choices = list(self.QUESTIONS[next_choice]["choices"].keys())
        self.user_choices[user_id]["choices_made"]["current_choice"] = next_choice
        return {"model_prompt": next_prompt, "choices": next_choices}


    def add_and_check_recent_questions_length(self, user_id, question):
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)

    def get_best_sentence_from_question_code(self, user_id, question_code):

        # get current_question prompt, and previous questions
        base_prompt = self.question_bank.loc[question_code]['question'] # question for this current node
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences']) # previous questions

        # potential rewritings of the prompt
        column = self.compassion_data[question_code].dropna() # this returns the vaierty of text, since compassion_data col name is question code
        
        # get best sentence
        question = self.get_best_sentence(column, prev_qs)

        # remove the sentence
        self.add_and_check_recent_questions_length(user_id, question)

        return question

    def get_count_by_question(self, user_id, user_session, question_code):
        if user_id not in self.nodes_count_by_user.keys():
            self.nodes_count_by_user[user_id] = {"A05": 0, "A11": 0, "B05": 0, "B09": 0, "C09": 0, "E04": 0}
        
        if question_code == "A05":
            if self.nodes_count_by_user[user_id][question_code] < 2:
                self.nodes_count_by_user[user_id][question_code] += 1
                return "trying_protocol_9"
            else:
                self.nodes_count_by_user[user_id][question_code] = 0
                return "user_found_no_compassion_to_child_after_3"

        elif question_code == "A11":
            if self.nodes_count_by_user[user_id][question_code] < 2:
                self.nodes_count_by_user[user_id][question_code] += 1
                return "trying_protocol_9"
            else:
                self.nodes_count_by_user[user_id][question_code] = 0
                return "come_back_tomorrow_before_main_node"

        elif question_code == "B05":
            if self.nodes_count_by_user[user_id][question_code] < 2:
                self.nodes_count_by_user[user_id][question_code] += 1
                return "trying_protocol_1_and_2"
            else:
                self.nodes_count_by_user[user_id][question_code] = 0
                return "come_back_tomorrow_before_main_node"

        elif question_code == "B09":
            if self.nodes_count_by_user[user_id][question_code] < 2:
                self.nodes_count_by_user[user_id][question_code] += 1
                return "trying_protocol_3_and_4_and_5"
            else:
                self.nodes_count_by_user[user_id][question_code] = 0
                return "come_back_tomorrow_before_main_node"
        
        elif question_code == "C09":
            if self.nodes_count_by_user[user_id][question_code] < 2:
                self.nodes_count_by_user[user_id][question_code] += 1
                return "trying_protocol_15"
            else:
                self.nodes_count_by_user[user_id][question_code] = 0
                return "sat_might_be_anti_social"

        elif question_code == "E04":
            if self.nodes_count_by_user[user_id][question_code] < 2:
                self.nodes_count_by_user[user_id][question_code] += 1
                return "trying_protocol_17_and_18"
            else:
                self.nodes_count_by_user[user_id][question_code] = 0
                return "come_back_tomorrow_before_main_node"

        return "main_node"
        """
        if question_code == "A05":
            if user_session.a05_count <= 3:
                user_session.a05_count += 1
                return "trying_protocol_9"
            else:
                user_session.a05_count = 0
                return "user_found_no_compassion_to_child_after_3"

        elif question_code == "A11":
            if user_session.a11_count <= 3:
                user_session.a11_count += 1
                return "trying_protocol_9"
            else:
                user_session.a11_count = 0
                return "come_back_tomorrow_before_main_node"

        elif question_code == "B05":
            if user_session.b05_count <= 3:
                user_session.b05_count += 1
                return "trying_protocol_1_and_2"
            else:
                user_session.b05_count = 0
                return "come_back_tomorrow_before_main_node"

        elif question_code == "B09":
            if user_session.b09_count <= 3:
                user_session.b09_count += 1
                return "trying_protocol_3_and_4_and_5"
            else:
                user_session.b09_count = 0
                return "come_back_tomorrow_before_main_node"
        
        elif question_code == "C09":
            if user_session.c09_count <= 3:
                user_session.c09_count += 1
                return "trying_protocol_15"
            else:
                user_session.c09_count = 0
                return "sat_might_be_anti_social"

        elif question_code == "E04":
            if user_session.e04_count <= 3:
                user_session.e04_count += 1
                return "trying_protocol_17_and_18"
            else:
                user_session.e04_count = 0
                return "come_back_tomorrow_before_main_node"

        return "main_node"

        """