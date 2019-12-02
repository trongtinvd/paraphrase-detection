from underthesea import pos_tag
from underthesea import word_tokenize

sen = 'kjkhigyf yêu cầu phát triển và ứng dụng công nghệ thông tin trong sản xuất kinh doanh và quản lý hướng tới mục tiêu nâng cao toàn diện năng lực cạnh tranh quốc gia coi đây là con đường ngắn nhất để Việt Nam tiến kịp các nước phát triển tiến cùng thời đại'
seg = word_tokenize(sen)
tag = pos_tag(sen)

tag2 = [i[1] for i in tag]

print('done')






