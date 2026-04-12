import http
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from backend.web_ingest import ingest_web_batch

url_dict = {
    "bài1": "https://loigiaihay.com/bai-4-do-dich-chuyen-va-quang-duong-di-duoc-trang-21-22-23-24-25-vat-li-10-ket-noi-tri-thuc-a106228.html",
    "bài2": "https://loigiaihay.com/bai-5-toc-do-va-van-toc-trang-26-27-28-29-vat-li-10-ket-noi-tri-thuc-a106242.html",
    "bài3": "https://loigiaihay.com/bai-7-do-thi-do-dich-chuyen-thoi-gian-trang-34-35-36-vat-li-10-ket-noi-tri-thuc-a106273.html",

    "bài4": "https://loigiaihay.com/bai-8-chuyen-dong-bien-doi-gia-toc-trang-37-38-39-vat-li-10-ket-noi-tri-thuc-a106285.html",

    "bài5": "https://loigiaihay.com/bai-9-chuyen-dong-thang-bien-doi-deu-trang-40-41-42-43-vat-li-10-ket-noi-tri-thuc-a106301.html",

    "bài6": "https://loigiaihay.com/bai-10-su-roi-tu-do-trang-44-45-46-vat-li-10-ket-noi-tri-thuc-a106306.html",

    "bài7": "https://loigiaihay.com/bai-12-chuyen-dong-nem-trang-49-50-51-52-53-54-vat-li-10-ket-noi-tri-thuc-a106326.html",

    "bài8": "https://loigiaihay.com/bai-13-tong-hop-va-phan-tich-luc-can-bang-luc-trang-56-57-58-59-vat-li-10-ket-noi-tri-thuc-a106341.html",

    "bài9": "https://loigiaihay.com/bai-14-dinh-luat-1-newton-trang-60-61-62-vat-li-10-ket-noi-tri-thuc-a106354.html",

    "bài10": "https://loigiaihay.com/bai-15-dinh-luat-2-newton-trang-63-64-65-66-vat-li-10-ket-noi-tri-thuc-a106362.html",

    "bài11": "https://loigiaihay.com/bai-16-dinh-luat-3-newton-trang-67-68-vat-li-10-ket-noi-tri-thuc-a106370.html",

    "bài12": "https://loigiaihay.com/bai-17-trong-luc-va-luc-cang-trang-69-70-71-vat-li-10-ket-noi-tri-thuc-a106380.html",

    "bài13": "https://loigiaihay.com/bai-18-luc-ma-sat-trang-72-73-74-75-76-vat-li-10-ket-noi-tri-thuc-a106390.html",

    "bài14": "https://loigiaihay.com/bai-19-luc-can-va-luc-nang-trang-77-78-79-vat-li-10-ket-noi-tri-thuc-a106399.html",

    "bài15": "https://loigiaihay.com/bai-20-mot-so-vi-du-ve-cach-giai-cac-bai-toan-thuoc-phan-dong-luc-hoc-trang-80-81-82-vat-li-10-ket-noi-tri-thuc-a106406.html",

    "bài16": "https://loigiaihay.com/giai-vat-li-10-bai-21-trang-83-84-85-ket-noi-tri-thuc-a107497.html",

    "bài17": "https://loigiaihay.com/giai-vat-li-10-bai-23-trang-91-92-93-94-95-ket-noi-tri-thuc-a107572.html",

    "bài18": "https://loigiaihay.com/giai-vat-li-10-bai-24-trang-96-97-98-ket-noi-tri-thuc24-a107575.html",

    "bài19": "https://loigiaihay.com/giai-vat-li-10-bai-25-trang-99-100-101-ket-noi-tri-thuc-a107578.html", 

    "bài20": "https://loigiaihay.com/giai-vat-li-10-bai-26-trang-102-103-104-105-ket-noi-tri-thuc-a107581.html",

    "bài21": "https://loigiaihay.com/giai-vat-li-10-bai-27-trang-106-107-108-ket-noi-tri-thuc-a107584.html",  
    "bài22": "https://loigiaihay.com/giai-vat-li-10-bai-28-trang-110-111-112-ket-noi-tri-thuc-a107588.html",

    "bài23": "https://loigiaihay.com/giai-vat-li-10-bai-29-trang-113-114-115-ket-noi-tri-thuc-a107595.html",

    "bài24": "https://loigiaihay.com/giai-vat-li-10-bai-30-trang-116-117-118-ket-noi-tri-thuc-a107597.html",

    "bài25": "https://loigiaihay.com/giai-vat-li-10-bai-31-trang-120-121-122-ket-noi-tri-thuc-a107599.html",

    "bài26": "https://loigiaihay.com/giai-vat-li-10-bai-32-trang-123-124-125-126-ket-noi-tri-thuc-a107602.html",

    "bài27": "https://loigiaihay.com/giai-vat-li-10-bai-33-trang-128-129-130-ket-noi-tri-thuc-a107605.html",

    "bài28": "https://loigiaihay.com/giai-vat-li-10-bai-34-trang-131-132-133-134-135-ket-noi-tri-thuc-a107612.html",

    "bài29": "https://loigiaihay.com/giai-vat-li-12-bai-1-trang-6-7-8-ket-noi-tri-thuc-a156372.html",

    "bài30": "https://loigiaihay.com/giai-vat-li-12-bai-2-trang-10-11-12-ket-noi-tri-thuc-a156375.html",

    "bài31": "https://loigiaihay.com/giai-vat-li-12-bai-3-trang-15-16-17-ket-noi-tri-thuc-a156381.html",

    "bài32": "https://loigiaihay.com/giai-vat-li-12-bai-4-trang-20-21-22-ket-noi-tri-thuc-a156834.html",

    "bài33": "https://loigiaihay.com/giai-vat-li-12-bai-5-trang-24-25-26-ket-noi-tri-thuc-a156836.html",

    "bài34": "https://loigiaihay.com/giai-vat-li-12-bai-6-trang-27-28-29-ket-noi-tri-thuc-a156838.html",

    "bài35": "https://loigiaihay.com/giai-vat-li-12-bai-7-trang-30-31-32-ket-noi-tri-thuc-a156841.html",

    "bài36": "https://loigiaihay.com/giai-vat-li-12-bai-8-trang-34-35-36-ket-noi-tri-thuc-a157885.html",

    "bài37": "https://loigiaihay.com/giai-vat-li-12-bai-9-trang-37-38-39-ket-noi-tri-thuc-a157888.html",

    "bài38": "https://loigiaihay.com/giai-vat-li-12-bai-10-trang-41-42-43-ket-noi-tri-thuc-a157889.html",

    "bài39": "https://loigiaihay.com/giai-vat-li-12-bai-11-trang-45-46-47-ket-noi-tri-thuc-a157891.html",

    "bài40": "https://loigiaihay.com/giai-vat-li-12-bai-12-trang-48-49-50-ket-noi-tri-thuc-a157893.html",

    "bài41": "https://loigiaihay.com/giai-vat-li-12-bai-13-trang-52-53-54-ket-noi-tri-thuc-a157896.html",

    "bài42": "https://loigiaihay.com/giai-vat-li-12-bai-14-trang-56-57-58-ket-noi-tri-thuc-a160066.html",

    "bài43": "https://loigiaihay.com/giai-vat-li-12-bai-15-trang-61-62-63-ket-noi-tri-thuc-a160072.html",

    "bài44": "https://loigiaihay.com/giai-vat-li-12-bai-16-trang-66-67-68-ket-noi-tri-thuc-a160084.html",

    "bài45": "https://loigiaihay.com/giai-vat-li-12-bai-17-trang-72-73-74-ket-noi-tri-thuc-a160094.html",

    "bài46": "https://loigiaihay.com/giai-vat-li-12-bai-18-trang-78-79-80-ket-noi-tri-thuc-a161519.html",

    "bài47": "https://loigiaihay.com/giai-vat-li-12-bai-19-trang-82-83-84-ket-noi-tri-thuc-a161540.html",

    "bài48": "https://loigiaihay.com/giai-vat-li-12-bai-20-trang-86-87-88-ket-noi-tri-thuc-a161815.html",

    "bài49": "https://loigiaihay.com/giai-vat-li-12-bai-21-trang-91-92-93-ket-noi-tri-thuc-a161818.html",

    "bài50": "https://loigiaihay.com/giai-vat-li-12-bai-22-trang-96-97-98-ket-noi-tri-thuc-a161829.html",    

    "bài51": "https://loigiaihay.com/giai-vat-li-12-bai-23-trang-104-105-106-ket-noi-tri-thuc-a161835.html",

    "bài52": "https://loigiaihay.com/giai-vat-li-12-bai-24-trang-114-115-116-ket-noi-tri-thuc-a161842.html",

    "bài53": "https://loigiaihay.com/giai-vat-li-12-bai-25-trang-119-120-121-ket-noi-tri-thuc-a161852.html",

}
ingest_web_batch(url_dict, force=False)