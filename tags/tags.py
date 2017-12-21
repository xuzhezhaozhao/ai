#! /usr/bin/env python
# -*-coding:utf-8 -*-


def delteduplicated(iterable):
    uniq = list()
    for x in iterable:
        if x not in uniq:
            uniq.append(x)
    return uniq


def load_taginfo_dict(inputfile):
    taginfo = {}
    for index, line in enumerate(open(inputfile, 'r')):
        line = unicode(line, 'utf-8')
        if index == 0:
            # skip header
            continue
        tokens = line.strip().split('/')
        try:
            tagid = int(tokens[0])
            tagname = reduce(lambda x, y: x+y, tokens[1:-1])
            taginfo[tagid] = tagname.replace(' ', '_')
        except Exception:
            pass

    return taginfo


def load_soso_taginfo_dict(inputfile):
    taginfo = {}
    for index, line in enumerate(open(inputfile, 'r')):
        line = unicode(line, 'utf-8')
        tokens = line.strip().split('\t')
        try:
            tagname = tokens[0]
            tagid = int(tokens[1])
            taginfo[tagid] = tagname.replace(' ', '_')
        except Exception:
            pass

    return taginfo
