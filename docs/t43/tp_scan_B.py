import ast,sys
src=open(sys.argv[1]).read(); tree=ast.parse(src)
funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
classes={n.name for n in tree.body if isinstance(n,ast.ClassDef)}
consts=set()
for n in tree.body:
    if isinstance(n,(ast.Assign,ast.AnnAssign)):
        for t in (n.targets if isinstance(n,ast.Assign) else [n.target]):
            for x in ast.walk(t):
                if isinstance(x,ast.Name): consts.add(x.id)
todo=['cmd_ground_truth_one'];seen=set();refs={}
while todo:
    f=todo.pop()
    if f in seen or f not in funcs: continue
    seen.add(f)
    for x in ast.walk(funcs[f]):
        if isinstance(x,ast.Name):
            if x.id in funcs: todo.append(x.id)
            elif x.id in classes or x.id in consts: refs.setdefault(x.id,set()).add(f)
print("functions:",sorted(seen))
print("unhashed module-level names used:",{k:sorted(v) for k,v in refs.items()})
