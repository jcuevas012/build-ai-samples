# Imports
from dotenv import load_dotenv
from anthropic import Anthropic
import json
import concurrent.futures
import re
from textwrap import dedent
from statistics import mean
from promt_evaluator import PromptEvaluator
# Client Initialization and helper functions

load_dotenv()

client = Anthropic()
model = "claude-haiku-4-5"


def add_user_message(messages, text):
    user_message = {"role": "user", "content": text}
    messages.append(user_message)


def add_assistant_message(messages, text):
    assistant_message = {"role": "assistant", "content": text}
    messages.append(assistant_message)


def chat(messages, system=None, temperature=1.0, stop_sequences=[]):
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature,
        "stop_sequences": stop_sequences,
    }

    if system:
        params["system"] = system

    message = client.messages.create(**params)
    return message.content[0].text

# Report Builder
evaluator = PromptEvaluator(max_concurrent_tasks=1)

dataset = evaluator.generate_dataset(
    # Describe the purpose or goal of the prompt you're trying to test
    task_description="Write a compact, concise 1 day meal plan for single athlete",
    # Describe the different inputs that your prompt requires
    prompt_inputs_spec={
        "height": "The height of the athlete in cm",
        "weight": "The weight of the athlete in kg",
        "goal": "The goal of the athlete (e.g. lose weight, gain muscle, maintain weight)",
        "restrictions": "Any dietary restrictions the athlete has (e.g. vegetarian, gluten-free, etc.)",
    },
    output_file="dataset_prompt_engineering_techniques.json",
    num_cases=3,
)


# Define and run the prompt you want to evaluate, returning the raw model output
# This function is executed once for each test case
def run_prompt(prompt_inputs):
    prompt = f"""
     Generate a one day meal plan for an athlete that meet their specific goals and dietary restrictions.
   
     <athlete_information>
        - Height: {prompt_inputs['height']} 
        - Weight: {prompt_inputs['weight']}
        - Goal: {prompt_inputs['goal']}
        - Dietary Restrictions: {prompt_inputs['restrictions']}
     </athlete_information>
     
     Guidelines:
        1. Include accurate daily calorie amount
        2. Show protein, fat, and carb amounts  
        3. Specify when to eat each meal
        4. Use only foods that fit restrictions
        5. List all portion sizes in grams
        6. Keep budget-friendly if mentioned

     Here is a sample input and ideal output to guide you:
          <input_sample>
        height: 178 cm
        weight: 82 kg
        goal: gain muscle and strength
        restrictions: none
     </input_sample>

     <ideal_output_sample>
            Daily Nutrition Targets
            - **Calories:** 3,200 kcal
            - **Protein:** 164g (2.0g per kg body weight)
            - **Carbohydrates:** 384g (4.7g per kg)
            - **Fat:** 107g (1.3g per kg)

            ---

            ## Meal Schedule

            ### **BREAKFAST** | 7:00 AM
            *Oatmeal Power Stack*
            - Oats: 80g
            - Whole eggs: 3 large (150g)
            - Banana: 150g
            - Honey: 20g
            - Olive oil: 10ml

            | Macros | Calories | Protein | Carbs | Fat |
            |--------|----------|---------|-------|-----|
            | | 650 kcal | 22g | 78g | 20g |

            ---

            ### **MID-MORNING SNACK** | 10:00 AM
            *Greek Yogurt & Granola*
            - Greek yogurt (plain): 200g
            - Granola: 50g
            - Berries (mixed): 80g
            - Almonds: 25g

            | Macros | Calories | Protein | Carbs | Fat |
            |--------|----------|---------|-------|-----|
            | | 420 kcal | 20g | 42g | 16g |

            ---

            ### **LUNCH** | 1:00 PM
            *Chicken & Rice Bowl*
            - Grilled chicken breast: 200g
            - Brown rice (cooked): 250g
            - Broccoli: 150g
            - Avocado: 60g
            - Olive oil: 10ml

            | Macros | Calories | Protein | Carbs | Fat |
            |--------|----------|---------|-------|-----|
            | | 750 kcal | 48g | 85g | 18g |

            ---

            ### **PRE-WORKOUT** | 3:30 PM
            *Quick Energy Boost*
            - White rice cakes: 50g
            - Peanut butter: 30g
            - Apple: 150g

            | Macros | Calories | Protein | Carbs | Fat |
            |--------|----------|---------|-------|-----|
            | | 380 kcal | 10g | 48g | 15g |

            ---

            ### **POST-WORKOUT** | 5:30 PM
            *Protein Shake*
            - Whey protein powder: 40g
            - Whole milk: 300ml
            - Banana: 100g
            - Honey: 15g

            | Macros | Calories | Protein | Carbs | Fat |
            |--------|----------|---------|-------|-----|
            | | 420 kcal | 32g | 52g | 8g |

            ---

            ### **DINNER** | 7:30 PM
            *Lean Beef & Pasta*
            - Lean ground beef (93/7): 200g
            - Whole wheat pasta (cooked): 200g
            - Tomato sauce: 100g
            - Parmesan cheese: 20g
            - Olive oil: 10ml

            | Macros | Calories | Protein | Carbs | Fat |
            |--------|----------|---------|-------|-----|
            | | 650 kcal | 42g | 72g | 20g |

            ---

            ### **EVENING SNACK** | 9:30 PM
            *Casein Protein Bedtime*
            - Cottage cheese (2%): 150g
            - Almonds: 30g
            - Berries: 50g

            | Macros | Calories | Protein | Carbs | Fat |
            |--------|----------|---------|-------|-----|
            | | 280 kcal | 30g | 17g | 10g |

            ---

            ## Daily Totals
            | **Metric** | **Amount** |
            |-----------|-----------|
            | **Total Calories** | 3,200 kcal |
            |
     </idal_output_sample>

     The solution comprehensively addresses all mandatory requirements with clear daily calorie targets, detailed macronutrient breakdowns, and specific meal suggestions across all meal types. It exceeds the 150g protein minimum with 164g distributed strategically around workout timing. Budget-conscious protein sources (eggs, chicken, cottage cheese, Greek yogurt) are prominently featured. The only material issue is the incomplete final summary table in the output, though this appears to be a formatting error rather than a content deficiency. All secondary criteria are met or exceeded. The meal plan is practical, well-organized, and appropriately tailored for a muscle-building athlete.


    """

    messages = []
    add_user_message(messages, prompt)
    return chat(messages)

results = evaluator.run_evaluation(
    run_prompt_function=run_prompt, 
    dataset_file="dataset_prompt_engineering_techniques.json",
    extra_criteria="""The output should include:
    - Daily calorie intake
    - Macronutrient breakdown (carbs, protein, fats)
    - Specific meal suggestions for breakfast, lunch, dinner, and snacks"""
)

