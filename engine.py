import cv2
import numpy as np
import math
import os

# load our image, calculate area


for filename in os.listdir('inputImages'):
    try:
        print(filename)
        picture = 'inputImages/' + filename
        game = cv2.imread(picture)
        image_area =  game.shape[0] * game.shape[1]

        # load shape templates for comparison later, calculate contours

        diamond = cv2.imread('shapeTemplates/diamond.jpg')
        oval = cv2.imread('shapeTemplates/oval.jpg')
        squiggle = cv2.imread('shapeTemplates/squiggle.jpg')

        ret, diamondThresh = cv2.threshold(cv2.imread('shapeTemplates/diamond.jpg', 0), 127, 255, cv2.THRESH_BINARY_INV)
        ret, ovalThresh = cv2.threshold(cv2.imread('shapeTemplates/oval.jpg', 0), 127, 255, cv2.THRESH_BINARY_INV)
        ret, squiggleThresh = cv2.threshold(cv2.imread('shapeTemplates/squiggle.jpg', 0), 127, 255, cv2.THRESH_BINARY_INV)

        diamondContours, hierarchy = cv2.findContours(diamondThresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        ovalContours, hierarchy = cv2.findContours(ovalThresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        squiggleContours, hierarchy = cv2.findContours(squiggleThresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        # a function used later to find furthest distance from center

        def pythagorean(distanceXY):
            return ((distanceXY[0] ** 2) + (distanceXY[1] ** 2))

        # variables to hold our contours, and eventually, cards

        contours = []
        cards = []

        # process entire image with threshold and erosion in order to calculate contours

        gamegray = cv2.cvtColor(game, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gamegray, 150, 255, 0)
        kernel = np.ones((4,4), np.uint8)
        eroded = cv2.erode(thresh, kernel, iterations = 1)

        #calculate contours in our image, many of which may be noise, hence the 'dirty' designation

        contoursDirty, hierarchy = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # add only those contours of a certain size, thus removing noise

        for cnt in contoursDirty:
            area = cv2.contourArea(cnt)
            if (area/image_area > 0.01):
                contours.append(cnt)

        # this should equal the number of cards in the frame, else something went wrong

        # print "contours found:", len(contours)

        # here we try to rotate each card so it's level. this is a bit tedious and involves multiple steps:

        # first,  we do this by looking at the coordinates of each card's contours and
        # divide them up according to which cartesian quadrant they are in: if the centroid of the card is
        # position (0,0) in the coordinate plane, then we can tell which of the cartesian quadrants
        # a given point is located in by seeing if it lies below or above, left or right of this centroid.

        for (i, c) in enumerate(contours):
            quad1 = []
            quad2 = []
            quad3 = []
            quad4 = []

            M = cv2.moments(c)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            for (i, d) in enumerate(c):

                distance = [(d[0][0] - cx), (d[0][1] - cy)]
                if (distance[0] < 0) and (distance[1] < 0) :
                    quad1.append(distance)
                if (distance[0] < 0) and (distance[1] > 0) :
                    quad2.append(distance)
                if (distance[0] > 0) and (distance[1] > 0) :
                    quad3.append(distance)
                if (distance[0] > 0) and (distance[1] < 0) :
                    quad4.append(distance)

            # then, we sort the points in each quadrant by their distance from the origin, picking
            # the farthest one. this is the approximate location of each corner

            top_left = sorted(quad1, key = pythagorean, reverse = True)[0]
            bottom_left = sorted(quad2, key = pythagorean, reverse = True)[0]
            bottom_right = sorted(quad3, key = pythagorean, reverse = True)[0]
            top_right = sorted(quad4, key = pythagorean, reverse = True)[0]

            # here we use the midpoint between a pair of top and bottom corners and some trigonometry
            # to calculate how far off the card is from level.

            distance_left_midpoint_above_centroid = (top_left[1] + bottom_left[1])/-2
            distance_left_midpoint_from_center = (top_left[0] + bottom_left[0])/-2
            angle = math.degrees(math.atan(float(distance_left_midpoint_above_centroid)/float(distance_left_midpoint_from_center)))

            # then we crop the card, and rotate it using the information we gathered above,
            # then re-crop it.

            (x, y, w, h) = cv2.boundingRect(c)

            height_to_subtract = (abs(distance_left_midpoint_above_centroid) + 0.04 * h)
            width_to_subtract = int(w - math.cos(math.radians(angle)) * w + 0.09 * w)
            single_card = game[y:y+h, x:x+w]
            rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            rotated_image = cv2.warpAffine(single_card, rotation_matrix, (w,h))
            rotated_cropped = rotated_image[height_to_subtract:h-height_to_subtract, width_to_subtract:w-width_to_subtract]

            # finally, we add our cropped and rotated card to our collection

            cards.append(rotated_cropped)

        # now it's time to analyze our cards for shape, shape count, shape color, and shape texture.

        for (i, c) in enumerate(cards):
            #cardstring will store information on each card
            cardString = ''

            #orient the card so lies flat, not tall
            if c.shape[0] > c.shape[1]:
                c = cv2.transpose(c)

            # process each card so we can calculate contours

            graycard = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
            _,threshcard = cv2.threshold(graycard, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # calculate all contours, and just external contours

            cardContoursAll, hierarchy = cv2.findContours(threshcard, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cardContoursExternal, hierarchy = cv2.findContours(threshcard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # evaluate shape count using external contour count

            cardString += str(len(cardContoursExternal))

            # a crude but effective way to evaluate texture by comparing total contour
            # count to external count. Solid shapes have no internal contours so the
            # external contout count === total contour count. empty shapes have two
            # contours: the outside and inside edges of their borders. so the total
            # contour count === 2 * external count. Striped shapes have lots of internal
            # contours, so if the ration of internal to external is high, we decide they are striped.

            if len(cardContoursAll) == len(cardContoursExternal):
                cardString += 'sol'
            elif len(cardContoursAll) == (len(cardContoursExternal) * 2):
                cardString += 'emp'
            elif len(cardContoursAll) > (len(cardContoursExternal) * 2):
                cardString += 'str'

            # evaluate shape by comparing to template

            diamondMatch = cv2.matchShapes(diamondContours[0], cardContoursExternal[0], 3, 0.0)
            ovalMatch = cv2.matchShapes(ovalContours[0], cardContoursExternal[0], 3, 0.0)
            squiggleMatch = cv2.matchShapes(squiggleContours[0], cardContoursExternal[0], 3, 0.0)

            if diamondMatch < ovalMatch and diamondMatch < squiggleMatch:
                cardString += 'Diamond'
            elif ovalMatch < squiggleMatch:
                cardString += 'Oval'
            else:
                cardString += 'Squiggle'

            # evaluating color: we just take a pixel biopsy from the edge of one of our shapes
            # and look at its RGB values.

            pixelSample = c[cardContoursExternal[0][0][0][1], cardContoursExternal[0][0][0][0]]
            if pixelSample[0] > pixelSample[1] and pixelSample[0] > pixelSample[2]:
                cardString += 'Purple'
            elif pixelSample[1] > pixelSample[2]:
                cardString += 'Green'
            else:
                cardString += 'Red'

            # now we check our string against the image of our card to check for accuracy. might
            # later choose to encode each card as a class or object.

            shapeDirectory = "./savedShapes/" + cardString
            if (os.path.isdir(shapeDirectory)):
                cv2.imwrite(shapeDirectory + "/" + cardString  + str(i) + filename, c)

            else:
                cv2.imwrite("./savedShapes/other/" + cardString  + str(i) + filename, c)
    except Exception:
        print(Exception)
        continue
