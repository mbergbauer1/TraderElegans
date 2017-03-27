#-----------------------------------------------------------------------------------------------------------------------
file_path = 'C:\\Users\\mbergbauer\\Desktop\\NN\\TraderElegans\\CSV_M1\\'
file_ext = '.csv'
filename_EURUSD = 'DAT_ASCII_EURUSD_M1_'
filename_GBPUSD = 'DAT_ASCII_GBPUSD_M1_'
filename_USDCHF = 'DAT_ASCII_USDCHF_M1_'
start_y = 2000
end_y = 2017
end_m = 3
start_time = 80000
end_time = 120000
#-----------------------------------------------------------------------------------------------------------------------


files = [filename_EURUSD, filename_GBPUSD, filename_USDCHF]

if end_m is None:
    out_file = file_path + 'training_set_'+str(start_y)+'_'+str(end_y)+'_'+str(start_time)+'_'+str(end_time)+'.txt'
else:
    out_file = file_path + 'training_set_'+str(start_y)+'_'+str(end_y)+str(end_m).zfill(2)+'_'+str(start_time)+'_'+str(end_time)+'.txt'

out = open(out_file,'w')
EURUSD = {}
GBPUSD = {}
USDCHF = {}

for year in range(start_y, end_y+1):


        if year == end_y:
            for month in range(1, end_m+1):
                file_EURUSD = file_path + filename_EURUSD + str(year) + str(month).zfill(2) + file_ext
                file_GBPUSD = file_path + filename_GBPUSD + str(year) + str(month).zfill(2) + file_ext
                file_USDCHF = file_path + filename_USDCHF + str(year) + str(month).zfill(2) + file_ext
                f1 = open(file_EURUSD, 'r')
                f2 = open(file_GBPUSD, 'r')
                f3 = open(file_USDCHF, 'r')

                for line1, line2, line3 in zip(f1, f2, f3):
                    line1 = line1.replace(" ", "")
                    line2 = line2.replace(" ", "")
                    line3 = line3.replace(" ", "")
                    line1 = line1.split(';')
                    line2 = line2.split(';')
                    line3 = line3.split(';')

                    if start_time <= int((line1[0])[8:14]) <= end_time:
                        EURUSD.update({line1[0]: line1[1:(len(line1) - 1)]})
                    if start_time <= int((line2[0])[8:14]) <= end_time:
                        GBPUSD.update({line2[0]: line2[1:(len(line2) - 1)]})
                    if start_time <= int((line3[0])[8:14]) <= end_time:
                        USDCHF.update({line3[0]: line3[1:(len(line3) - 1)]})

        else:
            file_EURUSD = file_path + filename_EURUSD + str(year) + file_ext
            file_GBPUSD = file_path + filename_GBPUSD + str(year) + file_ext
            file_USDCHF = file_path + filename_USDCHF + str(year) + file_ext
            f1 = open(file_EURUSD,'r')
            f2 = open(file_GBPUSD,'r')
            f3 = open(file_USDCHF,'r')

            for line1, line2, line3 in zip(f1,f2,f3):
                line1 = line1.replace(" ", "")
                line2 = line2.replace(" ", "")
                line3 = line3.replace(" ", "")
                line1 = line1.split(';')
                line2 = line2.split(';')
                line3 = line3.split(';')

                if start_time <= int((line1[0])[8:14]) <= end_time:
                    EURUSD.update({line1[0]:line1[1:(len(line1)-1)]})
                if start_time <= int((line2[0])[8:14]) <= end_time:
                    GBPUSD.update({line2[0]:line2[1:(len(line2)-1)]})
                if start_time <= int((line3[0])[8:14]) <= end_time:
                    USDCHF.update({line3[0]:line3[1:(len(line3)-1)]})

for key in sorted(EURUSD):

    tmp = key + ',' + str(EURUSD.get(key)).rstrip()
    if key in GBPUSD:
        tmp = tmp + ',' + str(GBPUSD.get(key)).rstrip()
    else:
        continue
    if key in USDCHF:
        tmp = tmp + ',' + str(USDCHF.get(key)).rstrip()
    else:
        continue
    tmp = tmp.replace ("'","")
    tmp = tmp.replace ("[","")
    tmp = tmp.replace ("]","")
    tmp = tmp.replace (" ","")
    tmp = tmp + '\n'
    out.write(tmp)