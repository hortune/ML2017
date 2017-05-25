with open('blend.csv','w') as fd:
    a = open('0','r').readlines()
    b = open('1','r').readlines()
    c = open('2','r').readlines()
    d = open('3','r').readlines()
    e = open('4','r').readlines()
    f = open('5','r').readlines()
    print('id,tags',file =fd)
    index = 0
    for x,y,z,g,q,qq in zip(a[1:],b[1:],c[1:],d[1:],e[1:],f[1:]):
        x = x.replace('"','').split(',')[1].split()
        y = y.replace('"','').split(',')[1].split()
        z = z.replace('"','').split(',')[1].split()
        g = g.replace('"','').split(',')[1].split()
        q = q.replace('"','').split(',')[1].split()
        qq = qq.replace('"','').split(',')[1].split()
        
        res = x+y+z+q+qq
        ans = []
        for i in g:
            if i not in ans:
                ans.append(i)
        for i in res:
            if i not in ans and res.count(i)>=3:
                ans.append(i)
        if len(ans) == 0:
            print(index,",\""," ".join(res),"\"",sep='',file=fd)
        else:
            print(index,",\""," ".join(ans),"\"",sep='',file=fd)
        index+=1
