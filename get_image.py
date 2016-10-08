import urllib2
import time

amt = 10000
img_path = "img/"
url = "http://xk.autoisp.shu.edu.cn:8080/Login/GetValidateCode?%20%20+%20GetTimestamp()"

for i in xrange(2312, amt, 1):
    r = urllib2.urlopen(url)
    f = open(img_path + str(i) + ".jpg", "wb")
    f.write(r.read())
    f.close()
    time.sleep(5)
