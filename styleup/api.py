import requests
import json
import time
import os
from decouple import config

BASE_URL = "https://api.luan.tools/api/tasks/"

# Here is our API key - remove it
#HEADERS = {
#    'Authorization': config('API_KEY'),
#    'Content-Type': 'application/json'}


# API-Function to create one picture
def send_task_to_dream_api(style_id, prompt, pic_width, pic_height, i, HEADERS, target_img_path=None):
    """
    Send requests to the dream API.
    prompt is the text prompt.
    style_id is which style to use (a mapping of ids to names is in the docs).
    target_img_path is an optional path to an image to influence the generation.
    """

    # Step 1) make a POST request to https://api.luan.tools/api/tasks/
    post_payload = json.dumps({
        "use_target_image": bool(target_img_path)
    })
    post_response = requests.request(
        "POST", BASE_URL, headers=HEADERS, data=post_payload)

    print(post_response.status_code)

    # Step 2) skip this step if you're not sending a target image otherwise,
    # upload the target image to the url provided in the response from the previous POST request.
    if target_img_path:
        target_image_url = post_response.json()["target_image_url"]
        with open(target_img_path, 'rb') as f:
            fields = target_image_url["fields"]
            fields ["file"] = f.read()
            requests.request("POST", url=target_image_url["url"], files=fields)

    # Step 3) make a PUT request to https://api.luan.tools/api/tasks/{task_id}
    # where task id is provided in the response from the request in Step 1.
    task_id = post_response.json()['id']
    task_id_url = f"{BASE_URL}{task_id}"
    put_payload = json.dumps({
        "input_spec": {
            "style": style_id,
            "prompt": prompt,
            "target_image_weight": 0.1,
            "width": pic_width,
            "height": pic_height
    }})
    # Testing why it does not work anymore #
    print("task_id:", task_id)
    print("task-id_url:", task_id_url)
    print(HEADERS, put_payload)
    requests.request(
        "PUT", task_id_url, headers=HEADERS, data=put_payload)

    # Step 4) Keep polling for images until the generation completes
    while True:
        response_json = requests.request(
            "GET", task_id_url, headers=HEADERS).json()

        state = response_json["state"]

        if state == "completed":
            r = requests.request(
                "GET", response_json["result"])
            # Path to save it
            os.chdir('pic-results')
            with open(f"image-{i}.jpg", "wb") as image_file:
                image_file.write(r.content)
            print("image saved successfully :)")
            os.chdir('..')
            break

        elif state =="failed":
            print("generation failed :(")
            break

        time.sleep(6)

    # Step 5) Enjoy your beautiful artwork :3




####     PICTURE FORMAT    ####
# Define for landscape and vertical - 2 different
def format_picture(pic_format):
    if pic_format == "Landscape format":
        pic_width = 1500
        pic_height = 1000
    elif pic_format == "Portrait format":
        pic_width = 1000
        pic_height = 1500
    else:
        print("failure - picture size not found")
    return pic_width, pic_height

####     3-Style IDS     ####
# Get the three style IDs for user choosen Topic
#user_topic = "Mountains"
def get_style_id_list_for_topic(pic_theme):
    cat_ids_topic = {"Abstract" : [17, 19, 14],
                    "Portrait" : [17, 3, 19],
                    "Mountains" : [19, 17, 2],
                    "Beach" : [17, 19, 15],
                    "City" : [4, 7, 9],
                    "Fruit" : [19, 16, 17],
                    "Plants" : [6, 1, 16],
                    "Flowers" : [6, 19, 17]
                    }
    return cat_ids_topic[pic_theme]
#run it
#get_style_id_list_for_topic(pic_theme)


#Call it
#pic_width, pic_height = format_picture(pic_format)


# Run and try it
# get topic from User input on front-end
#pic_theme = "Abstract"
# get the output from our models
#output_models = "pale blue brown deep brown dull retro clean rectilinear"

#prompt = user_topic + " " + output_models
#styles_for_topic = get_style_id_list_for_topic(pic_theme)
#pic_format = "Landscap format" # from user interface
#pic_width, pic_height = format_picture(pic_format)
#num_pictures_to_create = 3


# Make a loop to create X pictures

def create_multiple_pictures(prompt, pic_width, pic_height, num_pictures_to_create,
                             styles_for_topic, HEADERS):
    print("Number of pictures to create: ",num_pictures_to_create)
    print("Prompt: ", prompt)
    print("List of styles: ", styles_for_topic)
    for i in range(num_pictures_to_create):
        print(f"Get picture-{i}")
        print("style:",styles_for_topic[i])
        style_id = styles_for_topic[i]
        send_task_to_dream_api(style_id, prompt, pic_width, pic_height, i, HEADERS, target_img_path=None)
        print(f"picture-{i} created")
    api_run_finished = True
    return api_run_finished

#create_multiple_pictures(prompt, pic_width, pic_height, num_pictures_to_create,
#                             styles_for_topic)


# Run to create one single picture
#send_task_to_dream_api(style_id, prompt, end_pic_name, pic_width, pic_height, target_img_path=None)





# Style catalog Ids - just for internal documentation - We give
# three styles per topic in this variable: styles_for_topic = [1,15,16]
style_catalog = {1: "synthwave",
                 2: "ukiyoe",
                 3: "no-style",
                 4: "steampunk",
                 5: "fantasy-art",
                 6: "vibrant",
                 7: "hd",
                 9: "psychic",
                 10: "dark-fantasy",
                 11: "mystical",
                 13: "baroque",
                 14: "etching",
                 15: "dali",
                 16: "wuhtercuhler",
                 17: "realistic",
                 19: "throwback"
}


if __name__ == '__main__':
    try:
        pass

    except:
        pass
