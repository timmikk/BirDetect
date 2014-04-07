import re
import os

def read_names_from_file(name_file, reg_exp):
    names = {}
    prog = re.compile(reg_exp)

    for line in name_file:

        matchObj = prog.search(line)
        if matchObj is None:
            continue
        key = matchObj.group('key').lower()
        value = matchObj.group('value').lower()

        if len(key) > 0:
            names[key] = value
        else:
            print 'Not name line: ' + line

    return names

def read_filenames(files, reg_exp):
    names = {}
    prog = re.compile(reg_exp)



    for f in files:
        basename = os.path.basename(f)

        matchObj = prog.search(basename)

        if matchObj is None:
            print 'No match. reg_exp: ' + reg_exp + ' file: ' + basename
            continue
        key = matchObj.group('key').lower()

        if len(key) > 0:
            names[key] = os.path.abspath(f)
        else:
            print 'Not name line: ' + f

    return names


def rename_files(new_names, files, dry_run):
    for key in files.keys():
        f = files[key]

        if not new_names.has_key(key):
            continue

        file_ext = os.path.splitext(f)[1]
        dirname = os.path.dirname(f)

        new_name = new_names[key] + file_ext
        new_f = os.path.join(dirname, new_name)

        num = 2
        while os.path.exists(new_f):
            print 'File exists: ' + new_f
            new_name = new_names[key] + '_' + str(num) +file_ext
            new_f = os.path.join(dirname, new_name)
            num += 1
        else:
            if not dry_run:
                os.rename(f, new_f)
            print 'rename ' + f + ' -> ' + new_f

def rename(names_file, path, name_req_exp, file_reg_exp, dry_run):
    file_obj = open(names_file, 'r')
    names = read_names_from_file(file_obj, name_req_exp)
    files = [ os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) ]
    #files = [ f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) ]
    f_names = read_filenames(files, file_reg_exp)
    rename_files(names, f_names, dry_run)



#def rename_files(name_file, file_folder, name_req_exp, file_req_exp):

#name_file = '/media/timo/Tavara/Ohjelmointi/BirDetect/birds/bird_metadata/All_The_Bird_Songs_of_Britain_and_Europe_2.txt'
#reg_exp = '^0?(?P<key>[1-9]+)\s+(?<Pvalue>.+)\s.*'
#name_reg_exp = '^0?(?P<key>\d+)\s+(?P<value>\S+)\s'

#file_obj = open(name_file, 'r')
#names = read_names_from_file(file_obj, reg_exp)

#for key in names.keys():
#    print 'key=' + key + ' name=' + names[key]

#path = '/media/timo/Tavara/Ohjelmointi/BirDetect/birds/birds/All_The_Bird_Songs_of_Britain_and_Europe_2'

#files = [ os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) ]

#file_reg_exp = '^0?(?P<key>\d+)\D'
#f_names = read_filenames(files, reg_exp)

#for key in f_names.keys():
#    print 'key=' + key + ' file=' + f_names[key]

#rename_files(names, f_names, )

name_file = '/media/timo/Tavara/Ohjelmointi/BirDetect/birds/bird_metadata/Suomen_pollot.txt'
path = '/media/timo/Tavara/Ohjelmointi/BirDetect/birds/birds/Suomen_pollot'

#European ja Africa
name_reg_exp = '^0?(?P<key>\d+)\s+(?P<value>\S+)\s'
file_reg_exp = '^0?(?P<key>\d+)\D'

#Linu cd
#name_reg_exp = '(?P<key>\S+)\s+(?P<value>\S+)\s'
#file_reg_exp = '^(?P<key>\w*)'

rename(name_file, path, name_reg_exp, file_reg_exp, False)