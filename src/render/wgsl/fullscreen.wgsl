struct FsOut {
  @builtin(position) pos: vec4f,
}

@vertex
fn vsFull(@builtin(vertex_index) vi: u32) -> FsOut {
  let p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0))[vi];
  var o: FsOut;
  o.pos = vec4f(p, 0.0, 1.0);
  return o;
}
