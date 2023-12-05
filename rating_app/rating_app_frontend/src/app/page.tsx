"use client";

import Image from 'next/image'
import { useEffect, useState } from "react"

export default function Home() {
  useEffect(() => {
    const url = new URL(window.location.href);
    const code = url.searchParams.get("code");
    console.log(code);
  });

  return (
    <main className="flex flex-col items-center justify-center py-5 px-5">
      <div className="flex flex-col items-center justify-center">
        <span className="text">
          We are about to redirect to Spotify so you can log in with your account (we do not see your password) and allow us to retrieve your top listened songs, your liked songs, and your playlists. Please click the button below to continue.
        </span>
        <a className="bg-green-500 hover:bg-green-700 text-white py-2 px-4 rounded mt-5" href="https://accounts.spotify.com/authorize?client_id=5d6a9e7e2e0a4c5c9a5a0b2f5b7b3a0e&response_type=code&redirect_uri=http://localhost:3000/retrieve&scope=user-top-read%20user-read-recently-played%20user-library-read%20playlist-read-private%20playlist-read-collaborative">
          Continue
        </a>
      </div>
    </main>
  )
}
