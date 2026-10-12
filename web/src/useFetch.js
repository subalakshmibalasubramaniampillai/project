import { useEffect, useState } from "react";
import { get } from "./api";

// Fetch one or more API paths; returns { data, error, loading }.
// `data` is keyed by the names given in `paths`.
const cache = new Map();

export function useApi(paths) {
  const key = JSON.stringify(paths);
  const [state, setState] = useState({ data: null, error: null, loading: true });

  useEffect(() => {
    let alive = true;
    const entries = Object.entries(paths);
    Promise.all(
      entries.map(([, path]) => {
        if (!cache.has(path)) cache.set(path, get(path));
        return cache.get(path);
      }),
    )
      .then((values) => {
        if (!alive) return;
        const data = Object.fromEntries(entries.map(([name], i) => [name, values[i]]));
        setState({ data, error: null, loading: false });
      })
      .catch((e) => {
        entries.forEach(([, path]) => cache.delete(path));
        if (alive) setState({ data: null, error: e.message, loading: false });
      });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  return state;
}
