import io
import random
import string
from datetime import datetime, timedelta

import numpy as np
from PIL import Image


def random_description():
    num_letters = np.random.randint(0, 100)
    description = ""
    # printing lowercase
    letters = string.ascii_lowercase
    description += "".join(random.choice(letters) for i in range(num_letters))

    # printing uppercase
    num_letters = np.random.randint(0, 100)
    letters = string.ascii_uppercase
    description += "".join(random.choice(letters) for i in range(num_letters))

    # printing letters
    num_letters = np.random.randint(0, 100)
    letters = string.ascii_letters
    description += "".join(random.choice(letters) for i in range(num_letters))

    # printing digits
    num_letters = np.random.randint(0, 100)
    letters = string.digits
    description += "".join(random.choice(letters) for i in range(num_letters))

    # printing punctuation
    num_letters = np.random.randint(0, 100)
    letters = string.punctuation
    description += "".join(random.choice(letters) for i in range(num_letters))
    return description


def random_image():
    w = np.random.randint(160, 300)
    h = np.random.randint(160, 300)
    img = np.random.randint(255, size=(w, h, 3), dtype=np.uint8)
    img = Image.fromarray(img)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    return img_bytes.getvalue()


def random_date(start, end):
    """
    This function will return a random datetime between two datetime
    objects.
    """
    start = datetime.strptime(start, "%m/%d/%Y %I:%M %p")
    end = datetime.strptime(end, "%m/%d/%Y %I:%M %p")
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = random.randrange(int_delta)
    return start + timedelta(seconds=random_second)
