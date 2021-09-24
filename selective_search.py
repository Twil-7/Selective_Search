import cv2
import numpy as np
import skimage.segmentation
import random
import skimage.feature


# Selective Search algorithm

# step 1: calculate the first fel_segment region
# step 2: calculate the neighbour couple
# step 3: calculate the similarity dictionary
# step 4: merge regions and calculate the second merged region
# step 5: obtain e target candidate regions by secondary screening


def intersect(a, b):
    if (a["min_x"] < b["min_x"] < a["max_x"] and a["min_y"] < b["min_y"] < a["max_y"]) or \
            (a["min_x"] < b["max_x"] < a["max_x"] and a["min_y"] < b["max_y"] < a["max_y"]) or \
            (a["min_x"] < b["min_x"] < a["max_x"] and a["min_y"] < b["max_y"] < a["max_y"]) or \
            (a["min_x"] < b["max_x"] < a["max_x"] and a["min_y"] < b["min_y"] < a["max_y"]):
        return True
    return False


def calc_similarity(r1, r2, size):

    sim1 = 0
    sim2 = 0
    for a, b in zip(r1["hist_c"], r2["hist_c"]):
        sim1 = sim1 + min(a, b)
    for a, b in zip(r1["hist_t"], r2["hist_t"]):
        sim2 = sim2 + min(a, b)
    sim3 = 1.0 - (r1["size"] + r2["size"]) / size
    rect_size = (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"])) * \
             (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    sim4 = 1.0 - (rect_size - r1["size"] - r2["size"]) / size
    similarity = sim1 + sim2 + sim3 + sim4

    return similarity


def merge_region(r1, r2, t):
    new_size = r1["size"] + r2["size"]
    r_new = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "hist_c": (
            r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
        "hist_t": (
            r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
        "labels": t
    }
    return r_new


# Step 1: Calculate the different categories segmented by felzenszwalb algorithm

def first_calc_fel_category(image, scale, sigma, min_size):

    fel_mask = skimage.segmentation.felzenszwalb(image, scale=scale, sigma=sigma, min_size=min_size)
    print('The picture has been segmented in these categories : ', np.max(fel_mask))   # 0-694 categories

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    # (250, 250)
    texture_img = skimage.feature.local_binary_pattern(gray_image, 8, 1.0)    # (250, 250)

    # fel_img = np.zeros((fel_mask.shape[0], fel_mask.shape[0], 3))
    # for i in range(np.max(fel_mask)):
    #     a = random.randint(0, 255)
    #     b = random.randint(0, 255)
    #     c = random.randint(0, 255)
    #     for j in range(fel_mask.shape[0]):
    #         for k in range(fel_mask.shape[1]):
    #             if fel_mask[j, k] == i:
    #                 fel_img[j, k, 0] = a
    #                 fel_img[j, k, 1] = b
    #                 fel_img[j, k, 2] = c
    #
    # cv2.namedWindow("image")
    # cv2.imshow('image', fel_img/255)
    # cv2.waitKey(0)
    # cv2.imwrite('felzenszwalb_img.jpg', fel_img)

    img_append = np.zeros((fel_mask.shape[0], fel_mask.shape[1], 4))  # (250, 250, 4)
    img_append[:, :, 0:3] = image
    img_append[:, :, 3] = fel_mask

    region = {}

    # calc the min_x、in_y、max_x、max_y、label in every category
    for y, i in enumerate(img_append):
        for x, (r, g, b, l) in enumerate(i):
            if l not in region:
                region[l] = {"min_x": 0xffff, "min_y": 0xffff, "max_x": 0, "max_y": 0, "labels": l}
            if region[l]["min_x"] > x:
                region[l]["min_x"] = x
            if region[l]["min_y"] > y:
                region[l]["min_y"] = y
            if region[l]["max_x"] < x:
                region[l]["max_x"] = x
            if region[l]["max_y"] < y:
                region[l]["max_y"] = y

    for k, v in list(region.items()):

        # calc the size feature in every category
        masked_color = image[:, :, :][img_append[:, :, 3] == k]
        region[k]["size"] = len(masked_color)

        # calc the color feature in every category
        color_bin = 6
        color_hist = np.array([])

        for colour_channel in (0, 1, 2):
           c = masked_color[:, colour_channel]
           color_hist = np.concatenate([color_hist] + [np.histogram(c, color_bin, (0.0, 255.0))[0]])

        color_hist = color_hist / sum(color_hist)
        region[k]["hist_c"] = color_hist

        # calc the texture feature in every category
        texture_bin = 10
        masked_texture = texture_img[:, :][img_append[:, :, 3] == k]
        texture_hist = np.histogram(masked_texture, texture_bin, (0.0, 255.0))[0]
        texture_hist = texture_hist / sum(texture_hist)
        region[k]["hist_t"] = texture_hist

    return region


# Step 2: Calculate the neighbour couple in the first fel_segment region

def calc_neighbour_couple(region):
    r = list(region.items())
    couples = []

    for cur, a in enumerate(r[:-1]):
        for b in r[cur + 1:]:
            if intersect(a[1], b[1]):
                couples.append((a, b))

    return couples


# Step 3: Calculate the sim_dictionary in the neighbour couple

def calc_sim_dictionary(couple, total_size):

    sim_dictionary = {}

    for (ai, ar), (bi, br) in couple:
        sim_dictionary[(ai, bi)] = calc_similarity(ar, br, total_size)

    return sim_dictionary


# step 4: merge the small regions and calculate the second merged region

def second_calc_merge_category(sim_dictionary, region,  total_size):

    while sim_dictionary != {}:
        i, j = sorted(sim_dictionary.items(), key=lambda i: i[1])[-1][0]
        t = max(region.keys()) + 1.0

        region[t] = merge_region(region[i], region[j], t)
        key_to_delete = []
        for k, v in list(sim_dictionary.items()):
            if (i in k) or (j in k):
                key_to_delete.append(k)
        for k in key_to_delete:
            del sim_dictionary[k]

        for k in [a for a in key_to_delete if a != (i, j)]:
            n = k[1] if k[0] in (i, j) else k[0]
            sim_dictionary[(t, n)] = calc_similarity(region[t], region[n], total_size)

    return region


# step 5: obtain the target candidate regions by secondary screening

def calc_candidate_box(second_region, total_size):
    category = []
    for k, r in list(second_region.items()):
        category.append({'rect': (r['min_x'], r['min_y'], r['max_x'], r['max_y']), 'size': r['size']})

    candidate_box = set()
    for r in category:
        if r['rect'] in candidate_box:
            continue

        if r['size'] > total_size / 4:
            continue

        if r['size'] < total_size / 36:
            continue

        x1, y1, x2, y2 = r['rect']

        if (x2-x1) == 0 or (y2-y1) == 0:
            continue

        if (y2-y1) / (x2-x1) > 1.5 or (x2-x1) / (y2-y1) > 1.5:
            continue

        candidate_box.add(r['rect'])

    return candidate_box


img = cv2.imread('/home/archer/CODE/PF/162.jpg')
total_size = img.shape[0] * img.shape[1]
print('The shape of the image is : ', img.shape)    # (250, 250, 3)

first_region = first_calc_fel_category(img, scale=20, sigma=0.9, min_size=10)
print('first segment categories: ', len(first_region))

neighbour_couple = calc_neighbour_couple(first_region)
print('first neighbour_couple : ', len(neighbour_couple))

sim_dictionary = calc_sim_dictionary(neighbour_couple, total_size)

second_region = second_calc_merge_category(sim_dictionary, first_region, total_size)
print('second merge categories: ', len(second_region))

candidate_box = calc_candidate_box(second_region, total_size)
print('the candidate box we got by the selective search algorithm ： ')

flag = 1
for (x1, y1, x2, y2) in candidate_box:
    select_img = img[y1:y2, x1:x2]
    print(x1, y1, x2, y2)
    # cv2.namedWindow("select_image")
    # cv2.imshow("select_image", select_img)
    # cv2.waitKey(0)
    img_path ='/home/archer/CODE/PF/selective/' + str(flag) + '.jpg'
    cv2.imwrite(img_path, select_img)
    flag = flag + 1
