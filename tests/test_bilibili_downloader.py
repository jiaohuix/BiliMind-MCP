import os
import yt_dlp



class BilibiliDownloader:
    def __init__(self, output_dir: str = "downloads"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def download_audio(self, video_url: str) -> dict:
        """Download audio from Bilibili video"""
        output_path = os.path.join(self.output_dir, "%(id)s.%(ext)s")
        
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best',
            'outtmpl': output_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
            'quiet': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_id = info.get("id")
            audio_path = os.path.join(self.output_dir, f"{video_id}.mp3")

        return {
            'file_path': audio_path,
            'title': info.get("title"),
            'duration': info.get("duration", 0),
            'cover_url': info.get("thumbnail"),
            'video_id': video_id,
        }

    def download_video(self, video_url: str) -> str:
        """Download video from Bilibili"""
        output_path = os.path.join(self.output_dir, "%(id)s.%(ext)s")
        
        ydl_opts = {
            'format': 'bv*[ext=mp4]/bestvideo+bestaudio/best',
            'outtmpl': output_path,
            'quiet': True,
            'merge_output_format': 'mp4',
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_id = info.get("id")
            video_path = os.path.join(self.output_dir, f"{video_id}.mp4")

        return video_path

    def delete_file(self, file_path: str) -> bool:
        """Delete downloaded file"""
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False



if __name__ == "__main__":
    downloader = BilibiliDownloader()

    # # Download audio
    # audio_info = downloader.download_audio("https://www.bilibili.com/video/BV1xx411c7Xg")
    # print(f"Downloaded audio to: {audio_info['file_path']}")

    # # Download video
    # video_path = downloader.download_video("https://www.bilibili.com/video/BV1xx411c7Xg")
    # print(f"Downloaded video to: {video_path}")

    # # Delete file
    # downloader.delete_file(video_path)

    # url = "https://www.bilibili.com/video/BV11b4y1k7Y1?buvid=XUED6F9E1F21F98F1DC594A09116C21F8118D&from_spmid=search.search-result.0.0&is_story_h5=false&mid=V7gJxWRMpyPTZQqUMoP2rw%3D%3D&plat_id=116&share_from=ugc&share_medium=android&share_plat=android&share_session_id=d667f3e4-3c04-4861-a871-9eee0157e5c0&share_source=WEIXIN&share_tag=s_i&spmid=united.player-video-detail.0.0&timestamp=1741280420&unique_k=yQxNUf5&up_id=280987672&vd_source=6c90d268af3c1c349edae8f2df7eb06b&spm_id_from=333.788.videopod.sections"
    url = "https://www.bilibili.com/video/BV1z65TzuE94/?spm_id_from=333.337.search-card.all.click&vd_source=6c90d268af3c1c349edae8f2df7eb06b"
    audio_info = downloader.download_audio(url)
    print(f"Downloaded audio to: {audio_info['file_path']}")

