# general functions for main file


from styleup.color_detection import final_color_list


def string_output(final_color_names, final_adj_style_list):
    # combine to one list
    final_color_names.extend(final_adj_style_list)

    res_string = ""
    for word in final_color_names:
        res_string = res_string + " " + word

    return res_string
