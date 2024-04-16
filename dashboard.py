import cv2
import numpy as np
import json
import os
import math
from sklearn.linear_model import LinearRegression


class Dasboard():
    def __init__(self):

        self.vids_path = "beeVideos"
        v = os.listdir(self.vids_path)
        self.videos = []
        for vv in v:
            if vv[-4::].lower() in [".mkv", ".avi", ".mov"]:
                self.videos.append(vv)

        self.currentVideo = self.videos[20]# 9
        self.currentVideo = "a_queen_IMG_2398.mov"

        self.cap = cv2.VideoCapture(os.path.join(self.vids_path, self.currentVideo))

        ret, self.vid_img = self.cap.read()

        self.firstCallback = True

        self.annotColors = {"Bounding Boxes": (255,0,0), "Pose":(200,50,0), "Bee Segment":(0,255,0), "Frame":(0,0,255)}
        

        self.button_callbacks = [[],[],[],[]]
        self.dragCallbacks = [[], []]

        self.dragSpot = [-1,-1]

        self.dragging = False

        self.thumbnailOffset = 0
        self.vidNameOffset = 0 

        self.jetmap = True

        self.frameNum = 0

        self.masksShown = {"Bounding Boxes":True, "Pose":True, "Bee Segment":True, "Frame":True}


        self.vidImgType = ""

        # self.make_dashboard_high_level()
        self.make_dashboard_video()

        self.manageKeys()


    def manageKeys(self):
        k = -1
        while k < 0:
            k = cv2.waitKey(100)


    def mouseEvent(self, event, x, y, flags, param):
        if event == 4:
            if abs(self.dragSpot[0]-x) + abs(self.dragSpot[1]-y) < 20 and not self.dragging:
                for b in self.button_callbacks:
                    for a in b:
                        if a[0][0] < x < a[0][2] and a[0][1] < y < a[0][3]:
                            print("pressed button")
                            a[1](a[2])
            self.dragSpot = [-1, -1]

        elif event == 1:
            self.dragSpot = [x, y]
            self.dragging = False
            

        elif self.dragSpot[0] > 0 and ( abs(self.dragSpot[0]-x) + abs(self.dragSpot[1]-y) > 10 or self.dragging):

            for b in self.dragCallbacks:
                for a in b:
                    if a[0][0] < x < a[0][2] and a[0][1] < y < a[0][3]:
                        a[1]([self.dragSpot[0]-x, self.dragSpot[1]-y])
                        self.dragging = True
            self.dragSpot = [x, y]
        self.x = x
        self.y = y



    def rotate_point(self, pos, img_shape, rotation):
        """
        3->0
        0->1
        1->2
        2->3
        """
        if rotation == 0: #1
            return (pos[1], img_shape[1] - 1 - pos[0])
        elif rotation == 1: #2
            return (img_shape[1] - 1 - pos[0], img_shape[0] - 1 - pos[1])
        elif rotation == 2: #3 # This is actually 270 degrees clockwise
            return (img_shape[0] - 1 - pos[1], pos[0])
        else: # 3
            return pos  # No rotation


    def switchResType(self, button_name):
        print("switching to", button_name)
        self.final_img[10:460, 470:1300] = (255,255,255)

        if button_name == "Hive Coverage":
            self.addResImgGrouping(self.final_img)

        elif button_name == "Dist From Queen":
            self.addResImgDistance(self.final_img)

        elif button_name == "Bee Grouping":
            self.addResImgCount(self.final_img)



        elif button_name == "Color Map":
            self.jetmap = not self.jetmap

        cv2.imshow("img", self.final_img)

        # 

        # self.addResImgCount(final_img)

    def switchVidImgType(self, button_name):
        print(button_name)
        
        if button_name == "View Vid":
            self.make_dashboard_video()
        else:
            self.vidImgType = button_name

            self.addVidImgButtons(self.final_img)

            self.addVidImg(self.final_img)
            cv2.imshow("img", self.final_img)


    def switchVidThumbnail(self, vidName):
        print("switching to", vidName)
        self.currentVideo = vidName
        self.cap.release()
        self.cap = cv2.VideoCapture(os.path.join(self.vids_path, self.currentVideo))

        self.addVidImg(self.final_img)
        cv2.imshow("img", self.final_img)


    def scrollThumbnails(self, movement):

        
        self.thumbnailOffset += movement[1]
        self.addThumbnails(self.final_img, offset=self.thumbnailOffset)

        cv2.imshow("img", self.final_img)


    def addThumbnails(self, final_img, offset=0):
        bestViews = {'IMG_2395.mov': [55, 3], 'IMG_2397.mov': [118, 2], 'IMG_2391.mov': [72, 1], 'IMG_2280.mov': [317, 0], 'IMG_2185.mov': [117, 1], 'IMG_2394.mov': [32, 1], 'IMG_2287.mov': [81, 0], 'IMG_2390.mov': [220, 1], 'IMG_2279.mov': [132, 1], 'IMG_2396.mov': [48, 1], 'IMG_2393.mov': [215, 2]}

        self.button_callbacks[0] = []
        self.dragCallbacks[0] = []
        thumb_section = np.zeros((400, 500,3), np.uint8)

        namesW = 350
        tagH = 60

        thumb_section[:] = (200,200,200)


        for i, v in enumerate(self.videos):
            if v == self.currentVideo:
                pass

            cv2.rectangle(thumb_section, (10, tagH*i+10-offset), (490, tagH*i+tagH-offset), (0,0,0), 2)
            if v in bestViews:
                cv2.rectangle(thumb_section, (10, tagH*i+10-offset), (490, tagH*i+tagH-offset), (200,255,200), -1)
            else:
                cv2.rectangle(thumb_section, (10, tagH*i+10-offset), (490, tagH*i+tagH-offset), (255,255,255), -1)
            cv2.putText(thumb_section, v, (10, tagH*i+tagH-10-offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0),  1, cv2.LINE_AA)

            if thumb_section.shape[0] > tagH*i+10-offset > 0:
                self.button_callbacks[0].append([[10, tagH*i+10+900-thumb_section.shape[0]-10-offset, 490, tagH*i+tagH+900-thumb_section.shape[0]-10-offset], self.switchVidThumbnail, v])


        cv2.rectangle(thumb_section, (0,0), (thumb_section.shape[1]-1, thumb_section.shape[0]-1), (0,0,0), 1)

        cv2.line(final_img, (0, 900-thumb_section.shape[0]-20), (final_img.shape[1], 900-thumb_section.shape[0]-20), (0,0,0), 3)
        final_img[900-thumb_section.shape[0]-10:-10, 10:thumb_section.shape[1]+10] = thumb_section

        self.dragCallbacks[0].append([[10, 900-thumb_section.shape[0]-10, thumb_section.shape[1]+10, final_img.shape[0]-10], self.scrollThumbnails])
        # deal with later
        # thumbnail_names = os.listdir(os.path.join(self.vids_path, "thumbnails"))

        # for v in self.videos:
        #     if v + ".png" in thumbnail_names:
        #         img = cv2.imread(os.path.join())

            # cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def addResImgButtons(self, final_img):
        button_names = {"Hive Coverage": 0, "Bee Grouping": 1, "Dist From Queen" : 2, "Color Map": 3}

        self.button_callbacks[1] = []
        namesW = 270
        tagH = 60

        names_img = np.zeros((400, namesW, 3), np.uint8)
        names_img[:] = (255,200,200)


        for i, v in enumerate(button_names):
            if v == "idk":
                pass
            cv2.rectangle(names_img, (10, tagH*i+10), (namesW-10, tagH*i+tagH), (0,0,0), 2)
            cv2.rectangle(names_img, (10, tagH*i+10), (namesW-10, tagH*i+tagH), (255,255,255), -1)
            cv2.putText(names_img, v, (10, tagH*i+tagH-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0),  1, cv2.LINE_AA)

            self.button_callbacks[1].append([[200, tagH*i+20, 200+namesW, tagH*i+tagH+10], self.switchResType, v])

            # switchResType
        print(self.button_callbacks[1])

        cv2.rectangle(names_img, (0,0), (names_img.shape[1]-1, names_img.shape[0]-1), (0,0,0), 1)
        final_img[10:410, 200:200+namesW] = names_img

    def addResImgCount(self, final_img):
        bestViews = {'IMG_2395.mov': [55, 3], 'IMG_2397.mov': [118, 2], 'IMG_2391.mov': [72, 1], 'IMG_2280.mov': [317, 0], 'IMG_2185.mov': [117, 1], 'IMG_2394.mov': [32, 1], 'IMG_2287.mov': [81, 0], 'IMG_2390.mov': [220, 1], 'IMG_2279.mov': [132, 1], 'IMG_2396.mov': [48, 1], 'IMG_2393.mov': [215, 2]}

        resPlot = np.zeros((400, 400, 3),np.uint8)
        resPlot[:] = (200, 200, 200)

        path = "beeVideos"
        beeDetPath = "pose_det_feb16" # path of detected bees
        frameDetPath = "frame_res_feb21" # path of detected frames
        beeSegDetPath = "bee_res_feb22" # segmented bees
        vids = os.listdir(path)
        jsons = os.listdir(beeDetPath)

        ofcs = []
        pcs = [] 


        labelZone = np.zeros((50, 400,3), np.uint8)
        labelZone[:] = (255,255,255)
        cv2.putText(labelZone, "# of Bees", (140,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 1, cv2.LINE_AA)
        labelZone = cv2.rotate(labelZone, cv2.ROTATE_90_COUNTERCLOCKWISE)
        final_img[20:420, 560-50:560] = labelZone

        cv2.putText(final_img, "1000", (560-45,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

        cv2.putText(final_img, "0", (560-25, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)


        cv2.putText(final_img, "Percent of Frame Covered", (620,450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 1, cv2.LINE_AA)
        cv2.putText(final_img, "0", (560,440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

        cv2.putText(final_img, "100%", (560+380, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)


        for jsn in jsons: # loop through all videos + json files

            for vidNum, v in enumerate(vids):
                if v not in bestViews:
                    continue

                if jsn[0:-24] in v and (".mov" in v or ".mp4" in v): # get video


                    print("loading JSON")
                    with open(os.path.join(beeDetPath, jsn)) as json_file: # load bee json
                        vidInfoBee = json.load(json_file)

                    with open(os.path.join(frameDetPath, jsn[0:-24] + "feb21_frame_dets.json")) as json_file: # load frame json
                        vidInfoFrame = json.load(json_file)

                    vidInfoBeeSeg = {}

                    beeSegVid = cv2.VideoCapture(os.path.join(beeSegDetPath, jsn[0:-24] + "video.mkv"))

                    print("video", v)

                    frameViewed = {"vis": []}
                    cap = cv2.VideoCapture(os.path.join(path, v)) # begin video capture

                    fr = str(bestViews[v][0])
                    cap.set(1, bestViews[v][0])
                    beeSegVid.set(1, bestViews[v][0])
                    ret, img = cap.read() # read frame
                    ret, maskBee = beeSegVid.read()
                    m = cv2.cvtColor(maskBee, cv2.COLOR_BGR2GRAY)

                    mbb = maskBee.copy()
                    mbb[:,:,0] = 0
                    mbb[:,:,2] = 0

                    # img[m < 100] = 0

                    mask = np.zeros(img.shape, np.uint8) # make mask image

                    biggest = []
                    sBig = -1
                    if str(fr) in vidInfoFrame:
                        for det in vidInfoFrame[str(fr)]:
                            ar = np.array(det).transpose()
                            # print(ar[0].shape)
                            s = (np.max(ar[1]) - np.min(ar[1])) * (np.max(ar[0]) - np.min(ar[0]))
                            if s > sBig:
                                biggest = det
                                sBig = s
                    if len(biggest) > 0:
                        cv2.fillPoly(mask, [np.array(biggest)], (255,0,0)) # draw frame detection

                    # img[mask[:,:, 0] < 100] = 0
                    img = cv2.addWeighted(mask, 0.6, img, 1, 0, img) # combine mask with frame

                    img = cv2.addWeighted(mbb, 1, img, 1, 0, img) # combine mask with frame

                    totalCount = 0
                    onFrameCount = 0

                    sizes = []
                    for det in vidInfoBee[str(fr)]:
                        sizes.append((det[2]-det[0])*(det[3]-det[1]))
                    medSize = np.median(np.array(sizes))


                    for det in vidInfoBee[str(fr)]:
                            totalCount +=1


                            s = (det[2]-det[0])*(det[3]-det[1])
                            if s > medSize * 5:
                                # color = (0,255,255)
                                continue
                            
                            # set color of rectangle depending on whether the bee is on the frame
                            color = (0,0,255) # green (default)
                            x, y = int((det[0]+det[2])/2), int((det[1]+det[3])/2) 
                            if (0 < x < mask.shape[1] and 0 < y < mask.shape[0]) and mask[y, x][0] != 0:
                                color = (0,255,0) # red (on frame)
                                onFrameCount += 1
                            cv2.rectangle(img, (det[0], det[1]), (det[2], det[3]), color, 4) # draw bounding box

                    # print(onFrameCount, "on frame", "out of", totalCount)


                    mb = cv2.cvtColor(maskBee, cv2.COLOR_BGR2GRAY) 
                    mb[mb > 100] = 100
                    mb[mb < 100] = 0

                    mf = mask[:, :, 0]
                    mf[mf > 100] = 100
                    mf[mf < 100] = 0

                    total_frame = np.sum(mf>0)
                    total_bee = np.sum(mb>0)

                    comb = mf + mb
                    shared = np.sum(comb==200)
                    percentCover = shared/total_frame


                    # print("frame pixels", total_frame, "bee pixels", total_bee, "shared", shared, "percent", shared/total_frame)

                    if img.shape[0] < img.shape[1]:
                        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    img = cv2.resize(img, (200, 400))

                    final_img[20:420, 990:990+200] = img
                    

                    cap.release()

                    ofc = int(onFrameCount / 1000 * 400)
                    pc = int(400*percentCover)
                    pcs.append(pc)
                    ofcs.append(ofc)




                    model = LinearRegression()


                    resPlot[:] = (255,255,255)
                    if len(pcs) > 2:
                        x = np.array(pcs).reshape(-1, 1)
                        y = np.array(ofcs)

                        model.fit(x, y)
                        m = model.coef_[0]
                        b = model.intercept_
                        # y1 = m*0+b
                        # y2 = m*1+b
                        y1 = 0
                        y2 = int(400)
                        x1 = int((0-b)/m)
                        x2 = int((400-b)/m)

                        cv2.line(resPlot, (x1, resPlot.shape[0]-y1), (x2, resPlot.shape[0]-y2), (0,0,0), 2)
                    for pc, ofc in zip(pcs, ofcs):

                        cv2.rectangle(resPlot, (pc, resPlot.shape[0]-ofc),(pc, resPlot.shape[0]-ofc), (0,0,0), 10)



                    cv2.line(resPlot, (0,0), (0,resPlot.shape[0]), (0,0,0), 1)
                    cv2.line(resPlot, (0,resPlot.shape[0]-1), (resPlot.shape[1]-1,resPlot.shape[0]-1), (0,0,0), 1)
                    # final_img[20:420, 500:900] = resPlot
                    final_img[20:420, 560:560+400] = resPlot

                    cv2.imshow("img", final_img)
                    cv2.waitKey(100)

    def makeFrameSegImgs(self):
        bestViews = {'IMG_2395.mov': [55, 3], 'IMG_2397.mov': [118, 2], 'IMG_2391.mov': [72, 1], 'IMG_2280.mov': [317, 0], 'IMG_2185.mov': [117, 1], 'IMG_2394.mov': [32, 1], 'IMG_2287.mov': [81, 0], 'IMG_2390.mov': [220, 1], 'IMG_2279.mov': [132, 1], 'IMG_2396.mov': [48, 1], 'IMG_2393.mov': [215, 2]}
        for b in bestViews:
            with open(os.path.join(self.vids_path, b[0:-4] + "yolo_seg_frame.json")) as json_file: # load frame json
                vidInfoFrame = json.load(json_file)
            
            bee_cap = cv2.VideoCapture(os.path.join("bee_res_feb22", b[0:-4] + "video.mkv"))
            cap = cv2.VideoCapture(os.path.join(self.vids_path, b))
            for a in vidInfoFrame:
                if a != "classInfo":
                    cap.set(1, int(a))
                    bee_cap.set(1, int(a))
                    _, img = cap.read()
                    _, maskBee = bee_cap.read()

                    pts = []
                    for pt in vidInfoFrame[a][0][0:-1]:
                        pts.append(self.rotate_point(pt, img.shape, bestViews[b][1]))

                    for i in range(3-bestViews[b][1]):
                        # mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
                        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                        maskBee = cv2.rotate(maskBee, cv2.ROTATE_90_CLOCKWISE)
                  
                    # cv2.polylines(img, [np.array(pts)], True, (0,255,0), 15)

                    # cv2.imshow("img", img)
                    # cv2.imshow("mb", maskBee)

                    # cv2.imwrite(os.path.join("beeVideos/thumbnails", b[0:-4] + "_" + a + "_mask.png"), maskBee)
                    # cv2.imwrite(os.path.join("beeVideos/thumbnails", b[0:-4] + "_" + a + "_img.png"), img)

                    cv2.waitKey(1)

    def addVidImgButtons(self, final_img):
        modeTypes = ["Frame", "Masks", "Mask B&W", "Transform", "Distance", "Count", "View Vid"]

        self.button_callbacks[2] = []
        namesW = 180
        tagH = 55

        names_img = np.zeros((400, namesW, 3), np.uint8)
        names_img[:] = (255,200,200)


        for i, v in enumerate(modeTypes):
            if v == "idk":
                pass
            cv2.rectangle(names_img, (10, tagH*i+10), (namesW-10, tagH*i+tagH), (0,0,0), 2)
            cv2.rectangle(names_img, (10, tagH*i+10), (namesW-10, tagH*i+tagH), (255,255,255), -1)
            cv2.putText(names_img, v, (20, tagH*i+tagH-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0),  1, cv2.LINE_AA)
            self.button_callbacks[2].append([[10+final_img.shape[1]-10-namesW, final_img.shape[0]-10 -names_img.shape[0] + tagH*i+20, 10+final_img.shape[1]-10, final_img.shape[0]-10-names_img.shape[0]+ tagH*i+tagH+10], self.switchVidImgType, v])

        cv2.rectangle(names_img, (0,0), (names_img.shape[1]-1, names_img.shape[0]-1), (0,0,0), 1)
        final_img[final_img.shape[0]-10-names_img.shape[0]:final_img.shape[0]-10, final_img.shape[1]-10-namesW:final_img.shape[1]-10] = names_img

    def is_point_in_polygon(self, point, polygon):
        num_vertices = len(polygon)
        x, y = point
        inside = False

        p1x, p1y = polygon[0]
        for i in range(num_vertices + 1):
            p2x, p2y = polygon[i % num_vertices]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside


    def addVidImg(self, final_img):
        bestViews = {'IMG_2395.mov': [55, 3], 'IMG_2397.mov': [118, 2], 'IMG_2391.mov': [72, 1], 'IMG_2280.mov': [317, 0], 'IMG_2185.mov': [117, 1], 'IMG_2394.mov': [32, 1], 'IMG_2287.mov': [81, 0], 'IMG_2390.mov': [220, 1], 'IMG_2279.mov': [132, 1], 'IMG_2396.mov': [48, 1], 'IMG_2393.mov': [215, 2]}

        modeTypes = ["Frame", "Masks", "Mask B&W", "Transform", "Distance", "Count"]
        mode = self.vidImgType

        print(mode)
        jetmap = True


        vid_img = np.zeros((400, 800,3),np.uint8)

        if mode == "Frame":
            for i in os.listdir("beeVideos/thumbnails"):
                if self.currentVideo[0:-4] in i and "_img" in i:
                    vid_img = cv2.imread(os.path.join("beeVideos/thumbnails", i))

                    with open(os.path.join(self.vids_path, self.currentVideo[0:-4] + "yolo_seg_frame.json")) as json_file: # load frame json
                        vidInfoFrame = json.load(json_file)

                    _, test = self.cap.read()

                    pts = []
                    for a in vidInfoFrame:
                        if len(a) < 6 and "_" + a in i:

                            for pt in vidInfoFrame[a][0][0:-1]:
                                pts.append(self.rotate_point(pt, test.shape, bestViews[self.currentVideo][1]))

                    cv2.polylines(vid_img, [np.array(pts)], True, (0,255,0), 15)

                    vid_img = cv2.resize(vid_img, (800, 400))

                    break

        elif mode == "Masks":

            good = 0
            for i in os.listdir("beeVideos/thumbnails"):
                frame = ""
                if self.currentVideo[0:-4] in i and "_mask" in i:
                    a = i.split("_")
                    frame = a[-2]
                    mask_img = cv2.imread(os.path.join("beeVideos/thumbnails", i))
                    mask_img = cv2.resize(mask_img, (800, 400))
                    mask_img[:,:,0] = 0
                    mask_img[:,:,2] = 0
                    good += 1
                    break

            for i in os.listdir("beeVideos/thumbnails"):
                if self.currentVideo[0:-4] in i and "_img" in i and frame+"_" in i:
                    vid_img = cv2.imread(os.path.join("beeVideos/thumbnails", i))
                    vid_img = cv2.resize(vid_img, (800, 400))
                    good += 1
                    break
            if good == 2:
                vid_img = cv2.addWeighted(mask_img, 0.6, vid_img, 1, 0, vid_img) # combine mask with frame

        elif mode == "Mask B&W":
            for i in os.listdir("beeVideos/thumbnails"):
                if self.currentVideo[0:-4] in i and "_mask" in i:
                    vid_img = cv2.imread(os.path.join("beeVideos/thumbnails", i))
                    vid_img = cv2.resize(vid_img, (800, 400))

                    break

        elif mode == "Transform":
            for i in os.listdir("beeVideos/thumbnails"):
                if self.currentVideo[0:-4] in i and "_mask" in i:
                    mask_img = cv2.imread(os.path.join("beeVideos/thumbnails", i))
                    

                    
                    with open(os.path.join(self.vids_path, self.currentVideo[0:-4] + "yolo_seg_frame.json")) as json_file: # load frame json
                        vidInfoFrame = json.load(json_file)

                    _, test = self.cap.read()

                    pts = []
                    for a in vidInfoFrame:
                        if len(a) < 6 and "_" + a in i:

                            for pt in vidInfoFrame[a][0][0:-1]:
                                pts.append(self.rotate_point(pt, test.shape, bestViews[self.currentVideo][1]))

                    p = [0, 0, 0, 0]
                    for pt in pts:
                        print(pt, mask_img.shape)
                        if pt[1] < mask_img.shape[0]//2: # top half:
                            if pt[0] > mask_img.shape[1]//2: # right
                                p[3] = pt
                                print("top right")
                            else: # left
                                p[2] = pt
                                print("top left")
                        else: # bottom
                            if pt[0] > mask_img.shape[1]//2: # right
                                p[0] = pt
                                print("bottom right")
                            else: # left
                                p[1] = pt
                                print("bottom left")
                    print(p)
                    biggest = p[:]

                    coords = np.array(biggest, dtype=np.float32)
                    square_size = max(mask_img.shape[0], mask_img.shape[1])
                    dst = np.array([
                        [0, 0],
                        [square_size - 1, 0],
                        [square_size - 1, square_size - 1],
                        [0, square_size - 1]], dtype = "float32")

                    M = cv2.getPerspectiveTransform(coords, dst)
                    warped_image = cv2.warpPerspective(mask_img, M, (square_size, square_size))

                    rrFrame = cv2.resize(warped_image, (200, 200))


                    vid_img = cv2.resize(rrFrame, (800, 400))

                    break

        elif mode == "Distance":
            path = "beeVideos"
            dists = []
            angleDiffs = []
            for fname in os.listdir(path):
                    # print(fname)
                    if fname[-4::] == "json" and "seg" not in fname and self.currentVideo[0:-4] in fname:
                        print("found")
                        with open(os.path.join(path, fname)) as json_file:
                            allRects = json.load(json_file)

                        for frame in allRects:

                            queenSeen = False
                            queenCoords = [-1, -1]
                            sizes=  []
                            for r in allRects[frame]:
                                binary_str = bin(int(r[5]/2))[2:].zfill(2+1)
                                kpList = [not bool(int(digit)) for digit in binary_str]

                                invalid = False
                                for i, p in enumerate(kpList):
                                    if p == 0:
                                        invalid = True

                                classNum = int(r[4] // 2)
                                if classNum == 1:
                                    queenSeen = True
                                    queenCoords = [int((r[0]+r[2])/2), int((r[1]+r[3])/2), (r[6], r[7]), (r[8], r[9])]
                                elif not invalid: 
                                    s = max(abs(r[2]-r[1]), abs(r[3]-r[2]))
                                    sizes.append(s)

                            if not queenSeen:
                                continue
                            if len(sizes) == 0:
                                continue

                            self.cap.set(1, int(frame))
                            ret, img = self.cap.read()

                            s = sum(sizes) / len(sizes)

                            for r in allRects[frame]:

                                binary_str = bin(int(r[5]/2))[2:].zfill(2+1)
                                kpList = [not bool(int(digit)) for digit in binary_str]

                                invalid = False
                                for i, p in enumerate(kpList):
                                    if p == 0:
                                        invalid = True

                                if not invalid:     
                                    angleDiff = 0
                                    classNum = int(r[4] // 2)
                                    if classNum == 0:
                                        cv2.rectangle(img, (r[0], r[1]), (r[2], r[3]), (255,0,0), 2)
                                        c = [int((r[0]+r[2])/2), int((r[1]+r[3])/2)]
                                        
                                        dist = math.sqrt((queenCoords[0]-c[0])**2 + (queenCoords[1]-c[1])**2) / s
                                        dc = int(max(255-dist*80, 0))
                                        cv2.line(img, (c[0], c[1]), (queenCoords[0], queenCoords[1]), (0, dc,255-dc), 2)
                                        angleFacing = math.degrees(math.atan2((r[9] - r[7]), (r[8] - r[6])))


                                        angleRel1 = math.degrees(math.atan2((c[1] - queenCoords[2][1]), (c[0] - queenCoords[2][0])))
                                        angleRel2 = math.degrees(math.atan2((c[1] - queenCoords[3][1]), (c[0] - queenCoords[3][0])))

                                        angleDiff1 = ((angleFacing - angleRel1) % 180)
                                        angleDiff2 = ((angleFacing - angleRel2) % 180)

                                        angleDiff = min(angleDiff1, angleDiff2)


                                        if angleDiff > 180:
                                            print("this shouldnt happen")

                                        angleFacing = math.degrees(math.atan2((r[9] - r[7]), (r[8] - r[6])))
                                        cv2.rectangle(img, (int((r[8]+r[6])/2), int((r[9]+r[7])/2)), (int((r[8]+r[6])/2), int((r[9]+r[7])/2)), (0,0,255), 10)
                                        angleRel = (math.degrees(math.atan2((c[1] - (queenCoords[1])), (c[0] - queenCoords[0]))) + 180) % 360 # center
                                        
                                        angleDiff1 = min((angleFacing - angleRel1) % 360, 360 - ((angleFacing - angleRel1) % 360))
                                        angleDiff2 = min((angleFacing - angleRel2) % 360, 360 - ((angleFacing - angleRel2) % 360))
                                        angleDiff = min(angleDiff1, angleDiff2)

                                        angleDiff = min((angleFacing - angleRel) % 360, 360 - ((angleFacing - angleRel) % 360))


                                        if self.is_point_in_polygon((r[8], r[9]), (c, queenCoords[2], queenCoords[3])):
                                            angleDiff = 0


                                        angleDiffs.append(int(angleDiff))
                                        dists.append(dist)

                                    else:
                                        cv2.rectangle(img, (r[0], r[1]), (r[2], r[3]), (0,0,255), 2)
                                    cv2.arrowedLine(img, (r[6], r[7]), (r[8], r[9]), (0,int(255-angleDiff/180*255),int(angleDiff/180*255)), 2)

                            if img.shape[0] > img.shape[1]:
                                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            vid_img = cv2.resize(img, (800, 400))
                            break

                        if len(dists) > 0:
                            break

        elif mode == "Count":


            frameNum = 10


            bee_cap = cv2.VideoCapture(os.path.join("bee_res_feb22", self.currentVideo[0:-4] + "video.mkv"))
            # cap = cv2.VideoCapture(os.path.join(self.vids_path, self.currentVideo))
            bee_cap.set(1, frameNum)
            self.cap.set(1, frameNum)

            _, mask_img = bee_cap.read()
            _, vid_img = self.cap.read()

            mask_img[:,:,0] = 0
            mask_img[:,:,2] = 0



            mask = np.zeros(vid_img.shape, np.uint8)
            path = "frame_res_feb21"
            for fname in os.listdir(path):
                if fname[-4::] == "json" and self.currentVideo[0:-4] in fname:
                    with open(os.path.join(path, fname)) as json_file:
                        allPolys = json.load(json_file)
                    for p in allPolys[str(frameNum)]:
                        cv2.fillPoly(mask, pts = [np.array(p[0:-1])], color=(0,0,255))



            vid_img = cv2.addWeighted(mask, 0.5, vid_img, 1, 0, vid_img) # combine mask with frame


            vid_img = cv2.addWeighted(mask_img, 0.6, vid_img, 1, 0, vid_img) # combine mask with frame





            path = "pose_det_feb16"
            for fname in os.listdir(path):
                    # print(fname)
                    if fname[-4::] == "json" and "seg" not in fname and self.currentVideo[0:-4] in fname:
                        with open(os.path.join(path, fname)) as json_file:
                            allRects = json.load(json_file)
                        for r in allRects[str(frameNum)]:
                            cv2.rectangle(vid_img, (r[0], r[1]), (r[2], r[3]), (255,0,0), 2)




            if vid_img.shape[0] > vid_img.shape[1]:
                vid_img = cv2.rotate(vid_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            vid_img = cv2.resize(vid_img, (800, 400))

            

        final_img[final_img.shape[0]-400-10: final_img.shape[0]-10, final_img.shape[1]-800-200:final_img.shape[1]-200] = vid_img        


    def addResImgGrouping(self, final_img):
        bestViews = {'IMG_2395.mov': [55, 3], 'IMG_2397.mov': [118, 2], 'IMG_2391.mov': [72, 1], 'IMG_2280.mov': [317, 0], 'IMG_2185.mov': [117, 1], 'IMG_2394.mov': [32, 1], 'IMG_2287.mov': [81, 0], 'IMG_2390.mov': [220, 1], 'IMG_2279.mov': [132, 1], 'IMG_2396.mov': [48, 1], 'IMG_2393.mov': [215, 2]}

        resSection = np.zeros((400, 800, 3),np.uint8)


        jetmap = True
        imgs_added = 0
        coverImg = np.zeros((400,800,3),np.float32)
        for v in bestViews:
            for i in os.listdir("beeVideos/thumbnails"):
                if v[0:-4] in i and "_mask" in i:
                    mask_img = cv2.imread(os.path.join("beeVideos/thumbnails", i))
                    
                    b = i.split("_")
                    frame = b[-2]
                    

                    with open(os.path.join(self.vids_path, v[0:-4] + "yolo_seg_frame.json")) as json_file: # load frame json
                        vidInfoFrame = json.load(json_file)

                    test = mask_img.copy()
                    for i in range(3-bestViews[v][1]):
                        # mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
                        test = cv2.rotate(test, cv2.ROTATE_90_COUNTERCLOCKWISE)


                    pts = []
                    for a in vidInfoFrame:
                        if frame == a:
                            for pt in vidInfoFrame[a][0][0:-1]:
                                pts.append(self.rotate_point(pt, test.shape, bestViews[v][1]))

                    p = [0, 0, 0, 0]
                    for pt in pts:
                        if pt[1] < mask_img.shape[0]//2: # top half:
                            if pt[0] > mask_img.shape[1]//2: # right
                                p[3] = pt
                            else: # left
                                p[2] = pt
                        else: # bottom
                            if pt[0] > mask_img.shape[1]//2: # right
                                p[0] = pt
                            else: # left
                                p[1] = pt
                    
                    biggest = p[:]
                    coords = np.array(biggest, dtype=np.float32)
                    square_size = max(mask_img.shape[0], mask_img.shape[1])
                    dst = np.array([
                        [0, 0],
                        [square_size - 1, 0],
                        [square_size - 1, square_size - 1],
                        [0, square_size - 1]], dtype = "float32")

                    M = cv2.getPerspectiveTransform(coords, dst)
                    warped_image = cv2.warpPerspective(mask_img, M, (square_size, square_size))

                    rrFrame = cv2.resize(warped_image, (800, 400))

                    rrFrame = rrFrame.astype(np.float32)/255
                    coverImg += rrFrame
                    imgs_added += 1

                    coverImg_show = (coverImg/imgs_added*255).astype(np.uint8)
                    final_img[20:420, 500:1300] = coverImg_show
                    cv2.imshow("img", final_img)
                    cv2.waitKey(100)


        coverImg = (coverImg/imgs_added*255).astype(np.uint8)


        if self.jetmap:
            kde_values_normalized = cv2.normalize(coverImg, None, 0, 255, cv2.NORM_MINMAX)

            heatmap = np.uint8(kde_values_normalized)
            coverImg = cv2.applyColorMap(coverImg, cv2.COLORMAP_JET)

        final_img[20:420, 500:1300] = coverImg


    def addResImgDistance(self, final_img):
        resSection = np.zeros((400, 800, 3),np.uint8)

        jetmap = True
        dists = []
        angleDiffs = []

        total = 0

        labelZone = np.zeros((50, 400,3), np.uint8)
        labelZone[:] = (255,255,255)
        cv2.putText(labelZone, "Angle Relative to Queen", (50,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 1, cv2.LINE_AA)
        labelZone = cv2.rotate(labelZone, cv2.ROTATE_90_COUNTERCLOCKWISE)
        final_img[20:420, 560-50:560] = labelZone

        cv2.putText(final_img, "180", (560-40,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

        cv2.putText(final_img, "0", (560-25, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)



        cv2.putText(final_img, "Body Lengths From Queen", (620,450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 1, cv2.LINE_AA)
        cv2.putText(final_img, "0", (560,440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

        cv2.putText(final_img, "4", (560+380, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

        self.cap.release()

        plot = np.zeros((400, 400, 3), np.float32)
        path = "beeVideos"
        for fname in os.listdir(path):
            if fname[-4::] == "json" and "seg" not in fname:
                with open(os.path.join(path, fname)) as json_file:
                    allRects = json.load(json_file)

                    for v in os.listdir(path):
                        
                        # if v[-4::].lower() in [".jpg", ".png"]:
                        # if "mov" not in v:
                        #     continue
                        if True:

                            if v[0:-4] in fname[0:-6]:


                                if v[-4::].lower() in [".jpg", ".png"]:
                                    img = cv2.imread(os.path.join(path,v))
                                    # continue
                                else:
                                    cap = cv2.VideoCapture(os.path.join(path, v))


                                for frame in allRects:

                                    

                                    queenSeen = False
                                    queenCoords = [-1, -1]
                                    sizes=  []
                                    for r in allRects[frame]:
                                        binary_str = bin(int(r[5]/2))[2:].zfill(2+1)
                                        kpList = [not bool(int(digit)) for digit in binary_str]

                                        invalid = False
                                        for i, p in enumerate(kpList):
                                            if p == 0:
                                                invalid = True

                                        classNum = int(r[4] // 2)
                                        if classNum == 1:
                                            queenSeen = True
                                            queenCoords = [int((r[0]+r[2])/2), int((r[1]+r[3])/2), (r[6], r[7]), (r[8], r[9])]
                                        elif not invalid: 
                                            s = max(abs(r[2]-r[1]), abs(r[3]-r[2]))
                                            sizes.append(s)

                                    if not queenSeen:
                                        continue


                                    if len(sizes) == 0:
                                        continue

                                    if v[-4::].lower() not in [".jpg", ".png"]:
                                        cap.set(1, int(frame))
                                        ret, img = cap.read()



                                    s = sum(sizes) / len(sizes)

                                    dists = []
                                    angleDiffs = []

                                    for r in allRects[frame]:

                                        binary_str = bin(int(r[5]/2))[2:].zfill(2+1)
                                        kpList = [not bool(int(digit)) for digit in binary_str]

                                        invalid = False
                                        for i, p in enumerate(kpList):
                                            if p == 0:
                                                invalid = True

                                        if not invalid:     
                                            angleDiff = 0
                                            classNum = int(r[4] // 2)
                                            if classNum == 0:
                                                cv2.rectangle(img, (r[0], r[1]), (r[2], r[3]), (255,0,0), 2)
                                                c = [int((r[0]+r[2])/2), int((r[1]+r[3])/2)]
                                                # cv2.line(img, (c[0], c[1]), (queenCoords[0], queenCoords[1]), (255,255,255), 1)
                                                dist = math.sqrt((queenCoords[0]-c[0])**2 + (queenCoords[1]-c[1])**2) / s
                                                dc = int(max(255-dist*80, 0))
                                                cv2.line(img, (c[0], c[1]), (queenCoords[0], queenCoords[1]), (0, dc,255-dc), 2)
                                                angleFacing = math.degrees(math.atan2((r[9] - r[7]), (r[8] - r[6])))


                                                angleRel1 = math.degrees(math.atan2((c[1] - queenCoords[2][1]), (c[0] - queenCoords[2][0])))
                                                angleRel2 = math.degrees(math.atan2((c[1] - queenCoords[3][1]), (c[0] - queenCoords[3][0])))

                                                angleDiff1 = ((angleFacing - angleRel1) % 180)
                                                angleDiff2 = ((angleFacing - angleRel2) % 180)

                                                
                                                angleDiff = min(angleDiff1, angleDiff2)


                                                

                                                if angleDiff > 180:
                                                    print("wut")

                                                angleFacing = math.degrees(math.atan2((r[9] - r[7]), (r[8] - r[6])))
                                                cv2.rectangle(img, (int((r[8]+r[6])/2), int((r[9]+r[7])/2)), (int((r[8]+r[6])/2), int((r[9]+r[7])/2)), (0,0,255), 10)
                                                angleRel = (math.degrees(math.atan2((c[1] - (queenCoords[1])), (c[0] - queenCoords[0]))) + 180) % 360 # center

                                                angleDiff1 = min((angleFacing - angleRel1) % 360, 360 - ((angleFacing - angleRel1) % 360))
                                                angleDiff2 = min((angleFacing - angleRel2) % 360, 360 - ((angleFacing - angleRel2) % 360))
                                                angleDiff = min(angleDiff1, angleDiff2)

                                                angleDiff = min((angleFacing - angleRel) % 360, 360 - ((angleFacing - angleRel) % 360))



                                                # if self.is_point_in_polygon((r[8], r[9]), (c, queenCoords[2], queenCoords[3])):
                                                #     angleDiff = 0

                                                # angleDiff = np.random.random()*180
                                                angleDiffs.append(int(angleDiff))
                                                dists.append(dist)

                                                # cv2.putText(img, str(int(angleRel1)) + " " + str(int(angleRel2)), (c[0], c[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv2.LINE_AA)

                                                # cv2.putText(img, str(int(angleDiff1)) + " " + str(int(angleDiff2)) + " " + str(int(angleDiff)), c, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)

                                                # print(dist, angleFacing, angleRel, angleFacing - angleRel, angleDiff)
                                            else:
                                                cv2.rectangle(img, (r[0], r[1]), (r[2], r[3]), (0,0,255), 2)
                                            cv2.arrowedLine(img, (r[6], r[7]), (r[8], r[9]), (0,int(255-angleDiff/180*255),int(angleDiff/180*255)), 2)



                                    if img.shape[0] < img.shape[1]:
                                        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                    
                                    vid_img = cv2.resize(img, (200, 400))

                                    final_img[20:420, 500+600:1300] = vid_img


                                    # plot = np.zeros((400, 400, 3), np.float32)

                                    for dst0, dif in zip(dists, angleDiffs):
                                        dst = min(int(dst0*150),400)
                                        dif = min(int(dif*400/180), 400)

                                        plot[plot.shape[1]-dif-10:plot.shape[1]-dif, dst:dst+10] += 1

                                    
                                    
                                    if False:
                                        plot_show = plot.copy()
                                        plot_show = plot_show[:, :, 0]
                                        column_sums = plot_show.mean(axis=0)
                                        min_show = np.max(plot_show)/60
                                        target_sum = 100
                                        for x in range(plot_show.shape[1]):  # iterate over ccolumns
                                            col_sum = column_sums[x]
                                            # print(col_sum)

                                            if col_sum < min_show:
                                                plot_show[:, x] = 0
                                                continue 

                                            plot_show[:, x] *= 0.1/col_sum
                                        
                                        column_sums = plot_show.mean(axis=0)

                                        # print(np.max(plot_show), np.min(plot_show), np.mean(plot_show))
                                        plot_show = (plot_show*255.0/np.max(plot_show)).astype(np.uint8)
                                        ps = cv2.resize(plot_show, (400, 1))
                                        ps = cv2.resize(ps, (400, 20))
                                        # cv2.imshow("ps", ps)
                                        # cv2.imshow("pss", plot_show)
                                        plot_show = cv2.cvtColor(plot_show, cv2.COLOR_GRAY2BGR) 
                                    else:
                                    # plot_show = plot_show / x[:, np.newaxis]
                                        plot_show = (plot*255.0/np.max(plot)).astype(np.uint8)

                                    
                                    

                                    final_img[20:420, 560:560+400] = plot_show
                                    # cv2.imshow("pl", plot_show)




                                    total += len(angleDiffs)
                                    cv2.imshow("img", final_img)
                                    cv2.waitKey(1)
                                    # print(total)


                                if v[-4::].lower() in [".jpg", ".png"]:
                                    pass
                                else:
                                    cap.release()

        if self.jetmap:
            kde_values_normalized = cv2.normalize(plot_show, None, 0, 255, cv2.NORM_MINMAX)

            heatmap = np.uint8(kde_values_normalized)
            plot_show = cv2.applyColorMap(plot_show, cv2.COLORMAP_JET)

        self.cap = cv2.VideoCapture(os.path.join(self.vids_path, self.currentVideo))
        final_img[20:420, 560:560+400] = plot_show



    def make_dashboard_high_level(self):
        self.button_callbacks = [[],[],[],[]]
        self.dragCallbacks = [[], []]
        final_img = np.zeros((900, 1600,3), np.uint8)

        final_img[:] = (255,255,255)

        # self.makeFrameSegImgs()

        self.addThumbnails(final_img)

    
        self.addVidImg(final_img)

        self.addVidImgButtons(final_img)

        self.addResImgButtons(final_img)

        # self.addResImg(final_img)

        # self.addResImgDistance(final_img)

        # self.addResImgCount(final_img)

        cv2.imshow("img", final_img)

        self.final_img = final_img.copy()
        if self.firstCallback: # allow mouse events
            self.firstCallback = False
            print("cb")
            cv2.setMouseCallback("img", self.mouseEvent)


    def modMasksUsed(self, button_name):
        print(button_name)
        self.masksShown[button_name] = not self.masksShown[button_name]
        self.addVideo(self.final_img)
        cv2.imshow("img", self.final_img)


    def addVideo(self, final_img):

        img = self.vid_img.copy()

        # self.masksShown = {"Bounding Boxes":True, "Pose":True, "Bee Segment":True, "Frame":True}

        if self.masksShown["Bee Segment"]:
            bee_cap = cv2.VideoCapture(os.path.join("bee_res_feb22", self.currentVideo[0:-4] + "video.mkv"))
            # cap = cv2.VideoCapture(os.path.join(self.vids_path, self.currentVideo))
            bee_cap.set(1, self.frameNum)

            _, mask_img = bee_cap.read()

            mask_img[:,:,0] = 0
            mask_img[:,:,2] = 0

            img = cv2.addWeighted(mask_img, 0.6, img, 1, 0, img) # combine mask with frame

        if self.masksShown["Frame"]:
            mask = np.zeros(img.shape, np.uint8)
            path = "frame_res_feb21"
            for fname in os.listdir(path):
                if fname[-4::] == "json" and self.currentVideo[0:-4] in fname:
                    with open(os.path.join(path, fname)) as json_file:
                        allPolys = json.load(json_file)
                    if str(self.frameNum) in allPolys:
                        for p in allPolys[str(self.frameNum)]:
                            cv2.fillPoly(mask, pts = [np.array(p)], color=self.annotColors["Frame"])
            img = cv2.addWeighted(mask, 0.5, img, 1, 0, img) # combine mask with frame


        if self.masksShown["Bounding Boxes"] or self.masksShown["Pose"]:
            path = "pose_det_feb16"
            for fname in os.listdir(path):
                    # print(fname)
                    if fname[-4::] == "json" and "seg" not in fname and self.currentVideo[0:-4] in fname:
                        with open(os.path.join(path, fname)) as json_file:
                            allRects = json.load(json_file)
                        if str(self.frameNum) in allRects:
                            for r in allRects[str(self.frameNum)]:
                                if self.masksShown["Bounding Boxes"]:
                                    cv2.rectangle(img, (r[0],r[1]), (r[2],r[3]), self.annotColors["Bounding Boxes"],3)
                                if self.masksShown["Pose"]:
                                    if len(r) > 8: # keypoints available
                                        cv2.arrowedLine(img, (r[7], r[8]), (r[9], r[10]), self.annotColors["Pose"], 5)


        self.button_callbacks[3] = []
        if img.shape[0] > img.shape[1]:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.resize(img, (900, 450))
        
        final_img[100:550, 300:1200] = img

        tagH = 30
        legend_img = np.zeros((tagH*len(self.masksShown), 150,3), np.uint8)
        legend_img[:] = (255,255,255)

        for i, ln in enumerate(self.masksShown):
            color = (150,150,150)
            if self.masksShown[ln]:
                color = self.annotColors[ln]
            cv2.rectangle(legend_img, (10, tagH*i+2), (150-10, tagH*i+tagH-2), (0,0,0), 1)

            cv2.putText(legend_img, ln, (10, tagH*i+tagH-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,  1, cv2.LINE_AA)

            self.button_callbacks[3].append([[1200-legend_img.shape[1], 10+tagH*i+2, 1200-10, tagH*i+tagH-2+10], self.modMasksUsed, ln])

        final_img[10:legend_img.shape[0]+10, 1200-legend_img.shape[1]-10:1200-10] = legend_img


    def vidNameDrag(self, dragAmt):
        self.vidNameOffset += dragAmt[1]
        self.addVideoNames(self.final_img, offset=self.vidNameOffset)
        cv2.imshow("img", self.final_img)

    def vidNameCallback(self, vidName):
        print(vidName)
        self.frameNum = 0
        self.currentVideo = vidName
        self.cap.release()
        self.cap = cv2.VideoCapture(os.path.join(self.vids_path, self.currentVideo))
        _, self.vid_img = self.cap.read()

        self.addVideo(self.final_img)
        self.addVideoNames(self.final_img, offset=self.vidNameOffset)
        self.addFramePlot(self.final_img)
        cv2.imshow("img", self.final_img)


    def addVideoNames(self, final_img, offset=0):
        self.button_callbacks[1] = []
        namesW = 350
        tagH = 60

        names_img = np.zeros((650, namesW, 3), np.uint8)
        names_img[:] = (200,200,200)


        for i, v in enumerate(self.videos):
            t = 2
            if v == self.currentVideo:
                t=5
            cv2.rectangle(names_img, (10, tagH*i+10-offset), (340, tagH*i+tagH-offset), (0,0,0), t)
            cv2.rectangle(names_img, (10, tagH*i+10-offset), (340, tagH*i+tagH-offset), (255,255,255), -1)
            cv2.putText(names_img, v, (10, tagH*i+tagH-10-offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0),  1, cv2.LINE_AA)

            if 20+tagH*i-offset < 660:
                self.button_callbacks[1].append([[1600-namesW-25+10, 20+tagH*i-offset, 1600-25+340, 20+tagH*i+tagH-offset], self.vidNameCallback, v])

        cv2.rectangle(names_img, (0,0), (names_img.shape[1]-1, names_img.shape[0]-1), (0,0,0), 1)
        final_img[10:660, 1600-namesW-25:1600-25] = names_img
        self.dragCallbacks[0] = [[[1600-namesW-25, 10, 1600-25, 660], self.vidNameDrag]]


    def addImgPlayback(self, final_img):
        tagH = 30
        play_names = ["Reverse", "Play", "Fast Forward"]
        play_img = np.zeros((100, 600, 3), np.uint8)
        play_img[:] = (255,255,255)

        for i, ln in enumerate(legend_names):
            color = (0,0,0)
            if legend_names[ln]:
                color = self.annotColors[ln]

            cv2.putText(legend_img, ln, (10, tagH*i+tagH-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,  1, cv2.LINE_AA)

        final_img[10:legend_img.shape[0]+10, 900-legend_img.shape[1]-10:900-10] = legend_img


    def setFrameFromPlot(self, a):
        print("x", self.x)
        maxFrameNum = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame = int(self.x / 1600 * maxFrameNum)
        self.frameNum = frame
        self.cap.set(1, self.frameNum)
        _, self.vid_img = self.cap.read()
        self.addFramePlot(self.final_img)
        self.addVideo(self.final_img)
        cv2.imshow("img", self.final_img)


    def addFramePlot(self, final_img):
        framePlot = np.zeros((900-680, 1600, 3), np.uint8)
        framePlot[:] = (200,200,200)

        maxFrameNum = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        allRects = {}
        path = "pose_det_feb16"
        for fname in os.listdir(path):
            if fname[-4::] == "json" and "seg" not in fname and self.currentVideo[0:-4] in fname:
                with open(os.path.join(path, fname)) as json_file:
                    allRects = json.load(json_file)


        lastF = -2
        lastCount = -2
        # print(allRects)
        for f in range(maxFrameNum):
            if str(f) in allRects:
                count = len(allRects[str(f)])
                if f - lastF == 1:
                    cv2.line(framePlot, (int(lastF*1600/maxFrameNum), 220-int(lastCount*220/500)), (int(f*1600/maxFrameNum), 220-int(count*220/500)), (255,0,0), 2)
                else:
                    cv2.line(framePlot, (int(f*1600/maxFrameNum), 220-int(count*220/500)), (int(f*1600/maxFrameNum), 220-int(count*220/500)), (255,0,0), 2)
                lastCount = count
                lastF = f
        cv2.line(framePlot, (int(self.frameNum*1600/maxFrameNum), 0), (int(self.frameNum*1600/maxFrameNum), 220), (0,0,0), 1)



        self.button_callbacks[2] = [[[0, 680, 1600, 900], self.setFrameFromPlot, 0]]


        final_img[680::, :] = framePlot

        cv2.rectangle(final_img, (1430, 690), (1600, 730), (255,255,255), -1)
        cv2.putText(final_img, "Bee Count", (1450, 720), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),  1, cv2.LINE_AA)

    def processLeftButtons(self, button_name):
        print(button_name)
        if button_name == "Show All":
            self.make_dashboard_high_level()


    def addLeftButtons(self, final_img):
        leftButton_img = np.zeros((650, 250, 3), np.uint8)
        self.button_callbacks[0] = []
        tagH = 60
        leftButton_img[:] = (200,200,200)

        cv2.rectangle(leftButton_img, (0,0), (leftButton_img.shape[1]-1, leftButton_img.shape[0]-1), (0,0,0), 1)


        button_names = ["Add Boxes", "Add Segment", "Delete", "Show All"]
        for i, b in enumerate(button_names):
            cv2.rectangle(leftButton_img, (10, tagH*i+10), (240, tagH*i+tagH), (0,0,0), 2)
            cv2.rectangle(leftButton_img, (10, tagH*i+10), (240, tagH*i+tagH), (255,255,255), -1)
            cv2.putText(leftButton_img, b, (20, tagH*i+tagH-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0),  1, cv2.LINE_AA)
            self.button_callbacks[0].append([[10+10, tagH*i+10+10, 10+240, tagH*i+tagH+10], self.processLeftButtons, b])

        final_img[10:660, 10:260] = leftButton_img



    def make_dashboard_video(self):

        self.button_callbacks = [[],[],[],[]]
        self.dragCallbacks = [[], []]


        final_img = np.zeros((900, 1600,3), np.uint8)

        final_img[:] = (255,255,255)

        
        self.addVideo(final_img)

        self.addVideoNames(final_img)

        self.addFramePlot(final_img)

        self.addLeftButtons(final_img)


        cv2.imshow("img", final_img)

        self.final_img = final_img.copy()
        if self.firstCallback: # allow mouse events
            self.firstCallback = False
            cv2.setMouseCallback("img", self.mouseEvent)
        
if __name__ == "__main__":
    dashboard = Dasboard()
        