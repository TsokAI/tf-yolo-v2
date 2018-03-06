from __future__ import absolute_import, division, print_function
import numpy as np


def parse_xml(evalanno, frame_id):
    pass


def eval_frame(frame_id, detections):
    annotations = parse_xml(frame_id)

    fpre_rec = {}

    for classname in ['car', 'bus', 'van', 'others']:
        A = [obj for obj in annotations if obj['name'] == classname]
        D = [obj for obj in detections if obj['name'] == classname]

        na = len(A)
        nd = len(D)

        if na == 0 and nd == 0:
            fpre_rec[classname] = (None, None)
        elif na*nd == 0:
            fpre_rec[classname] = (0, 0)
        else:
            gtbb = np.array([x['bbox'] for x in A])
            diff = np.array([x['difficult'] for x in A]).astype(np.bool)
            dets = [False] * len(A)
            npos = sum(~diff)

            coef = np.array([x['confidence'] for x in D])
            prbb = np.array([[x['topleft']['x'], x['topleft']['y'],
                              x['bottomright']['x'], x['bottomright']['y']] for x in D])

            sorted_inds = np.argsort(-coef)
            coef = coef[sorted_inds]
            prbb = prbb[sorted_inds]

            tp = 0
            fp = 0

            for i in range(len(prbb)):
                bb = prbb[i]

                ixmin = np.maximum(gtbb[:, 0], bb[0])
                iymin = np.maximum(gtbb[:, 1], bb[1])
                ixmax = np.minimum(gtbb[:, 2], bb[2])
                iymax = np.minimum(gtbb[:, 3], bb[3])

                iw = np.maximum(ixmax - ixmin + 1, 0)
                ih = np.maximum(iymax - iymin + 1, 0)
                inter = iw * ih

                uni = ((bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) +
                       (gtbb[:, 2] - gtbb[:, 0] + 1) * (gtbb[:, 3] - gtbb[:, 1] + 1) - inters)

                overlaps = inter / uni
                jmax = np.argmax(overlaps)

                if overlaps[jmax] > 0.5:
                    if not diff[jmax]:
                        if not det[jmax]:
                            tp += 1
                            det[jmax] = 1
                        else:
                            fp += 1
                else:
                    fp += 1

            rec = tp / float(npos)
            pre = tp / (tp + fp)

            fpre_rec[classname] = (pre, rec)

    return fpre_rec
